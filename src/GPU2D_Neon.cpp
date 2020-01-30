#include "GPU2D.h"

#include "NDS.h"
#include "GPU.h"

#include <string.h>
#include <arm_neon.h>

#include <assert.h>

/*
    optimised GPU2D for aarch64 devices
    which are usually less powerful, but support the NEON vector instruction set

    Q&A:
        Why inline assembler instead of intrinsics?
            I would have loved to use intrinsics if GCC would produce good code for them.
            Sometimes it does, but once you cross the teritory of shrn/shrn2 and stuff 
            doing things like partial register writes the compiler output is just bad.
            Not even speaking of missing instructions such as ld1 with a set of four
            registers. 
        Why are scratch registers hard coded?
            For some stupid reason GCC only allows a maximum of 30 inline asm operands
            (and read/write operands count twice!!). I used to use leave ARM temp register 
            allocation to the compiler, until I hit the limit. Also Neon instructions 
            like ld1 or tbl require adjacent registers which the compiler can't really
            at all (manual register allocation seems to be broken/disabled with GCC 9.2.0),
            so there manual allocation is the only choice.

    "If you have 32 registers - use 32 registers!!111!!11"
        - me

*/

GPU2DNeon::GPU2DNeon(u32 num)
    : GPU2DBase(num)
{
}

void GPU2DNeon::Reset()
{
    GPU2DBase::Reset();

    BGExtPalStatus = 0;
}

void GPU2DNeon::DoSavestate(Savestate* file)
{
    GPU2DBase::DoSavestate(file);
}

void GPU2DNeon::SetDisplaySettings(bool accel)
{
    // OGL renderer is unsupported in conjunction with the Neon renderer
}

void GPU2DNeon::BGExtPalDirty(u32 base)
{
    BGExtPalStatus &= ~((u64)0xFFFFFFFF << (base * 16));
}

void GPU2DNeon::OBJExtPalDirty()
{
}

void GPU2DNeon::EnsurePaletteCoherent()
{
    u64 paletteUpdates = BGExtPalUsed & ~BGExtPalStatus;
    BGExtPalStatus |= paletteUpdates;
    BGExtPalUsed = 0;

    u8* base = GPU::Palette + (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset) * 256 * 2;

    while (paletteUpdates != 0)
    {
        int idx = __builtin_ctzll(paletteUpdates);

        u16* dst = (u16*)(base + idx * 256 * 2);
        if (Num)
        {
            u32 mapping = GPU::VRAMMap_BBGExtPal[idx >> 4];
            if (mapping & (1<<7))
                memcpy(dst, &GPU::VRAM_H[idx * 256 * 2], 256*2);
            else
                memset(dst, 0, 256*2);
        }
        else
        {
            u32 mapping = GPU::VRAMMap_ABGExtPal[idx >> 4];
            memset(dst, 0, 256*2);
            if (mapping & (1<<4))
                for (int i = 0; i < 256; i += 4)
                    *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_E[idx * 256 * 2 + i * 2];
            if (mapping & (1<<5))
                for (int i = 0; i < 256; i += 4)
                    *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_F[(idx * 256 * 2 & 0x3FFF) + i * 2];
            if (mapping & (1<<6))
                for (int i = 0; i < 256; i += 4)
                    *(u64*)&dst[i] |= *(u64*)&GPU::VRAM_G[(idx * 256 * 2 & 0x3FFF) + i * 2];
        }

        paletteUpdates &= ~((u64)1 << idx);
    }
}

void GPU2DNeon::DrawScanline(u32 line)
{
    u32* dst = &Framebuffer[256 * line];

    int n3dline = line;
    line = GPU::VCount;
    
    if (Num == 0 && (CaptureCnt & (1<<31)) && (((CaptureCnt >> 29) & 0x3) != 1))
        _3DLine = GPU3D::GetLine(n3dline);

    DrawScanline_BGOBJ(line);
    UpdateMosaicCounters(line);

    EnsurePaletteCoherent();
    bool arg = false;
    for (int i = 0; i < 256; i++)
    {
        u32 val = BGOBJLine[i + 8];
        u32 color1 = *(u16*)&GPU::Palette[(val & 0xFFFF) * 2];
        u8 r = (color1 & 0x001F) << 1;
        u8 g = (color1 & 0x03E0) >> 4;
        u8 b = (color1 & 0x7C00) >> 9;
        dst[i] = r | (g << 8) | (b << 16);
    }
    for (int i = 0; i < 256; i+=2)
    {
        u64 c = *(u64*)&dst[i];

        u64 r = (c << 18) & 0xFC000000FC0000;
        u64 g = (c << 2) & 0xFC000000FC00;
        u64 b = (c >> 14) & 0xFC000000FC;
        c = r | g | b;

        *(u64*)&dst[i] = c | ((c & 0x00C0C0C000C0C0C0) >> 6) | 0xFF000000FF000000;
    }
}

void GPU2DNeon::DrawScanline_BGOBJ(u32 line)
{
    {
        u128 backdrop = Num ? 0x200 : 0;
        backdrop |= 0x20000000;
        backdrop |= (backdrop << 32);
        backdrop |= (backdrop << 64);

        for (int i = 0; i < 256; i+=4)
            *(u128*)&BGOBJLine[i + 8] = backdrop;
    }

    if (DispCnt & 0xE000)
        CalculateWindowMask(line, &WindowMask[8], &OBJWindow[8]);
    else
        memset(WindowMask + 8, 0xFF, 256);

    switch (DispCnt & 0x7)
    {
    case 0: DrawScanlineBGMode<0>(line); break;
    case 1: DrawScanlineBGMode<1>(line); break;
    case 2: DrawScanlineBGMode<2>(line); break;
    case 3: DrawScanlineBGMode<3>(line); break;
    case 4: DrawScanlineBGMode<4>(line); break;
    case 5: DrawScanlineBGMode<5>(line); break;
    //case 6: DrawScanlineBGMode6(line); break;
    //case 7: DrawScanlineBGMode7(line); break;
    }
}

#define DoDrawBG(type, line, num) \
    { if ((BGCnt[num] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_##type<true>(line, num); else DrawBG_##type<false>(line, num); }

#define DoDrawBG_Large(line) \
    { if ((BGCnt[2] & 0x0040) && (BGMosaicSize[0] > 0)) DrawBG_Large<true>(line); else DrawBG_Large<false>(line); }

template<u32 bgmode>
void GPU2DNeon::DrawScanlineBGMode(u32 line)
{
    for (int i = 3; i >= 0; i--)
    {
        if ((BGCnt[3] & 0x3) == i)
        {
            if (DispCnt & 0x0800)
            {
                if (bgmode >= 3)
                    //DoDrawBG(Extended, line, 3)
                {}
                else if (bgmode >= 1)
                {}    //DoDrawBG(Affine, line, 3)
                else
                    DoDrawBG(Text, line, 3)
            }
        }
        if ((BGCnt[2] & 0x3) == i)
        {
            if (DispCnt & 0x0400)
            {
                if (bgmode == 5)
                    {}//DoDrawBG(Extended, line, 2)
                else if (bgmode == 4 || bgmode == 2)
                    {}//DoDrawBG(Affine, line, 2)
                else
                    DoDrawBG(Text, line, 2)
            }
        }
        if ((BGCnt[1] & 0x3) == i)
        {
            if (DispCnt & 0x0200)
            {
                DoDrawBG(Text, line, 1)
            }
        }
        if ((BGCnt[0] & 0x3) == i)
        {
            if (DispCnt & 0x0100)
            {
                if ((!Num) && (DispCnt & 0x8))
                   {} //DrawBG_3D();
                else
                    DoDrawBG(Text, line, 0)
            }
        }
        //if ((DispCnt & 0x1000) && NumSprites)
        //    InterleaveSprites(0x40000 | (i<<16));
    }
}

template<bool mosaic>
void GPU2DNeon::DrawBG_Text(u32 line, u32 bgnum)
{
    u16 bgcnt = BGCnt[bgnum];

    u32 tilesetaddr, tilemapaddr;
    u32 extpal, extpalslot;

    u16 xoff = BGXPos[bgnum];
    u16 yoff = BGYPos[bgnum] + line;

    if (bgcnt & 0x0040)
    {
        // vertical mosaic
        yoff -= BGMosaicY;
    }

    u32 widexmask = (bgcnt & 0x4000) ? 0x100 : 0;

    extpal = (DispCnt & 0x40000000);
    if (extpal) extpalslot = ((bgnum<2) && (bgcnt&0x2000)) ? (2+bgnum) : bgnum;

    if (Num)
    {
        tilesetaddr = ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((bgcnt & 0x1F00) << 3);
    }
    else
    {
        tilesetaddr = ((DispCnt & 0x07000000) >> 8) + ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((DispCnt & 0x38000000) >> 11) + ((bgcnt & 0x1F00) << 3);
    }

    u32 width = (bgcnt & 0x4000) ? 512 : 256;
    u32 height = (bgcnt & 0x8000) ? 512 : 256;
    u8* tilemapptr = GPU::GetBGCachePtr(Num, tilemapaddr, width * height * 2);
    u8* tilesetptr = GPU::GetBGCachePtr(Num, tilesetaddr, 1024 * (bgcnt & 0x80 ? 16 : 32));

    // adjust Y position in tilemap
    if (bgcnt & 0x8000)
    {
        tilemapptr += ((yoff & 0x1F8) << 3);
        if (bgcnt & 0x4000)
            tilemapptr += ((yoff & 0x100) << 3);
    }
    else
        tilemapptr += ((yoff & 0xF8) << 3);

#define loadTile(out, xoff) \
    "and w8, " xoff ", #0xF8\n" \
    "and w9, " xoff ", %w[widexmask]\n" \
    "add x8, %[tilemapptr], x8, lsr #2\n" \
    "lsl w9, w9, #3\n" \
    "ldrh " out ", [x8, x9]\n"
#define loadPixels4(out, n, tile) \
    "and w9, " tile ", 0x3FF\n" \
    "add x9, %[tilesetptr], x9, lsl #5\n" \
    "tst " tile ", #0x800\n" \
    "csel w8, %w[yoff], %w[yoffFlipped], eq\n" \
    "add x8, x8, x9\n" \
    "ld1 {" out ".s}[" n "], [x8]\n"
#define loadPixels8(out, n, tile) \
    "and w9, " tile ", 0x3FF\n" \
    "add x9, %[tilesetptr], x9, lsl #6\n" \
    "tst " tile ", #0x800\n" \
    "csel w8, %w[yoff], %w[yoffFlipped], eq\n" \
    "add x8, x8, x9\n" \
    "ld1 {" out ".d}[" n "], [x8]\n"
    // used to be called loadStuff, but that was too informal
#define loadMisc \
    "ld1 {v6.16b, v7.16b}, [%[windowMask]], #32\n" \
    \
    "ld4 {v20.16b, v21.16b, v22.16b, v23.16b}, [%[dst]]\n" \
    "add x8, %[dst], #272*4\n" \
    "ld4 {v24.16b, v25.16b, v26.16b, v27.16b}, [x8]\n" \
    "add x9, %[dst], #64\n" \
    "ld4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x9]\n"
#define weavePixels(primary0, primary1) \
    "add x10, %[dst], #(272*4)+64\n" \
    "ld4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x10]\n" \
    \
    /* where overriden replace/move first 16 pixels one layer down */ \
    "bif v24.16b, v20.16b, v2.16b\n" \
    "bif v25.16b, v21.16b, v2.16b\n" \
    "bif v26.16b, v22.16b, v2.16b\n" \
    "bif v27.16b, v23.16b, v2.16b\n" \
    \
    "bif v20.16b, v0.16b, v2.16b\n" \
    "bif v21.16b, " primary0 ".16b, v2.16b\n" \
    "bif v22.16b, %[paletteIndexSecondary].16b, v2.16b\n" \
    "bif v23.16b, %[bgnumBit].16b, v2.16b\n" \
    \
    "st4 {v24.16b, v25.16b, v26.16b, v27.16b}, [x8]\n" \
    \
    /* next 16 pixels */ \
    "bif v16.16b, v28.16b, v3.16b\n" \
    "bif v17.16b, v29.16b, v3.16b\n" \
    "bif v18.16b, v30.16b, v3.16b\n" \
    "bif v19.16b, v31.16b, v3.16b\n" \
    \
    "st4 {v20.16b, v21.16b, v22.16b, v23.16b}, [%[dst]]\n" \
    \
    "bif v28.16b, v1.16b, v3.16b\n" \
    "bif v29.16b, " primary1 ".16b, v3.16b\n" \
    "bif v30.16b, %[paletteIndexSecondary].16b, v3.16b\n" \
    "bif v31.16b, %[bgnumBit].16b, v3.16b\n" \
    \
    "st4 {v16.16b, v17.16b, v18.16b, v19.16b}, [x10]\n" \
    "st4 {v28.16b, v29.16b, v30.16b, v31.16b}, [x9]\n"
#define asmDefaultOutput \
    [dst] "+r" (dst), [windowMask] "+r" (windowMask), \
    [xoff] "+r" (xoff)
#define asmDefaultInput(tileshift) \
    [tilemapptr] "r" (tilemapptr), [tilesetptr] "r" (tilesetptr), \
    [widexmask] "r" (widexmask), \
    [xofftarget] "r" (xofftarget), \
    [yoff] "r" ((yoff & 0x7) << tileshift), [yoffFlipped] "r" ((7 - (yoff & 0x7)) << tileshift), \
    [bgnumBit] "w" (vdupq_n_u8(1 << bgnum)), [hflipMask] "w" (vdupq_n_u8(1 << 2)), \
    [paletteIndexSecondary] "w" (vdupq_n_u8(0))
#define asmDefaultScratch \
    "x8", "x9", "x10", "x11", "x12", "x13", \
    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", \
    "q16", "q17", "q18", "q19", "q20", "q21", "q22", "q23", \
    "q24", "q25", "q26", "q27", "q28", "q29", "q30", "q31", \
    "memory", "cc"
#define regExtpalUse(reg) \
    "mov w8, #1\n" \
    "ubfx " reg ", " reg ", #12, #4\n" \
    "lsl w8, w8, " reg "\n" \
    "orr %w[extpalsUsed], %w[extpalsUsed], w8\n"

    u32 localxoff = 8 - (xoff & 0x7);
    xoff &= ~0x7;

    u32* dst = &BGOBJLine[localxoff];
    u8* windowMask = &WindowMask[localxoff];
    u32 xofftarget = xoff + 256 + (localxoff == 8 ? 0 : 8);

    if (bgcnt & 0x80)
    {
        u32 palOffset;
        if (extpal)
            palOffset = extpalslot * 16 + (Num ? GPU::FastPalExtBOffset : GPU::FastPalExtAOffset);
        else
            palOffset = Num ? 2 : 0;
        u64 extpalsUsed = 0;

        asm volatile (
            "loopStart%=:\n"
                loadTile("w10", "%w[xoff]")
                "add w9, %w[xoff], #8\n"
                loadTile("w11", "w9")
                "add w9, %w[xoff], #16\n"
                loadTile("w12", "w9")
                "add w9, %w[xoff], #24\n"
                loadTile("w13", "w9")
                "add %w[xoff], %w[xoff], #32\n"

                "ubfx w8, w10, #8, #8\n"
                "dup v18.16b, w8\n"
                "ubfx w8, w11, #8, #8\n"
                "dup v19.16b, w8\n"
                loadPixels8("v0", "0", "w10")
                loadPixels8("v0", "1", "w11")
                regExtpalUse("w10")
                regExtpalUse("w11")
                "ext v16.16b, v18.16b, v19.16b, #8\n"

                "ubfx w8, w12, #8, #8\n"
                "dup v18.16b, w8\n"
                "ubfx w8, w13, #8, #8\n"
                "dup v19.16b, w8\n"
                loadPixels8("v1", "0", "w12")
                loadPixels8("v1", "1", "w13")
                regExtpalUse("w12")
                regExtpalUse("w13")
                "ext v17.16b, v18.16b, v19.16b, #8\n"

                loadMisc

                "and v18.16b, v16.16b, %[hflipMask].16b\n"
                "and v19.16b, v17.16b, %[hflipMask].16b\n"
                "rev64 v2.16b, v0.16b\n"
                "rev64 v3.16b, v1.16b\n"
                "cmeq v18.16b, v18.16b, #0\n"
                "cmeq v19.16b, v19.16b, #0\n"
                "ushr v4.16b, v16.16b, #4\n"
                "ushr v5.16b, v17.16b, #4\n"
                "bif v0.16b, v2.16b, v18.16b\n"
                "bif v1.16b, v3.16b, v19.16b\n"
                "and v6.16b, v6.16b, %[bgnumBit].16b\n"
                "and v7.16b, v7.16b, %[bgnumBit].16b\n"
                "cmeq v2.16b, v0.16b, #0\n"
                "cmeq v6.16b, v6.16b, #0\n"
                "cmeq v3.16b, v1.16b, #0\n"
                "cmeq v7.16b, v7.16b, #0\n"
                "and v4.16b, v4.16b, %[extpalMask].16b\n"
                "and v5.16b, v5.16b, %[extpalMask].16b\n"
                "orr v2.16b, v2.16b, v6.16b\n"
                "orr v3.16b, v3.16b, v7.16b\n"
                "add v4.16b, v4.16b, %[palOffset].16b\n"
                "add v5.16b, v5.16b, %[palOffset].16b\n"

                weavePixels("v4", "v5")

                "add %[dst], %[dst], #128\n"
                "cmp %w[xoff], %w[xofftarget]\n"
                "blo loopStart%=\n"

            "cmp %w[xoff], %w[xofftarget]\n"
            "sub %w[xoff], %w[xoff], #56\n"
            "sub %[dst], %[dst], #56*4\n"
            "sub %[windowMask], %[windowMask], #56\n"
            "bne loopStart%=\n"
            :
                asmDefaultOutput, [extpalsUsed] "+r" (extpalsUsed)
            :
                asmDefaultInput(3), [extpalMask] "w" (vdupq_n_u8(extpal ? 0xFF : 0)),
                [palOffset] "w" (vdupq_n_u8(palOffset))
            :
                asmDefaultScratch
        );

        if (extpal)
            BGExtPalUsed |= extpalsUsed << (extpalslot * 16);
    }
    else
    {
        asm volatile (
            "loopStart%=:\n"
                // load the first four tiles
                loadTile("w10", "%w[xoff]")
                "add w9, %w[xoff], #8\n"
                loadTile("w11", "w9")
                "add w9, %w[xoff], #16\n"
                loadTile("w12", "w9")
                "add w9, %w[xoff], #24\n"
                loadTile("w13", "w9")
                "add %w[xoff], %w[xoff], #32\n"

                "ubfx w8, w10, #8, #8\n"
                "dup v18.16b, w8\n"
                "ubfx w8, w11, #8, #8\n"
                "dup v19.16b, w8\n"
                loadPixels4("v0", "0", "w10")
                loadPixels4("v0", "1", "w11")
                "ext v16.16b, v18.16b, v19.16b, #8\n"

                "ubfx w8, w12, #8, #8\n"
                "dup v18.16b, w8\n"
                "ubfx w8, w13, #8, #8\n"
                "dup v19.16b, w8\n"
                loadPixels4("v0", "2", "w12")
                loadPixels4("v0", "3", "w13")
                "ext v17.16b, v18.16b, v19.16b, #8\n"

                loadMisc

                "shl v2.16b, v0.16b, #4\n" // unpack 4 bit indices
                "ushr v1.16b, v0.16b, #4\n"
                "ushr v18.16b, v16.16b, #4\n" // extract palette offset
                "ushr v19.16b, v17.16b, #4\n"
                "ushr v2.16b, v2.16b, #4\n"
                "and v16.16b, v16.16b, %[hflipMask].16b\n" // generate mask from h-flip bit
                "and v17.16b, v17.16b, %[hflipMask].16b\n"
                "zip1 v0.16b, v2.16b, v1.16b\n"
                "zip2 v1.16b, v2.16b, v1.16b\n"
                "cmeq v16.16b, v16.16b, #0\n"
                "cmeq v17.16b, v17.16b, #0\n"
                "rev64 v2.16b, v0.16b\n"
                "rev64 v3.16b, v1.16b\n"
                "shl v18.16b, v18.16b, #4\n"
                "shl v19.16b, v19.16b, #4\n"
                "bif v0.16b, v2.16b, v16.16b\n" // apply flipping
                "bif v1.16b, v3.16b, v17.16b\n"
                "and v6.16b, v6.16b, %[bgnumBit].16b\n" // prepare window mask
                "and v7.16b, v7.16b, %[bgnumBit].16b\n"
                "cmeq v2.16b, v0.16b, #0\n" // generate tile visibility mask
                "cmeq v3.16b, v1.16b, #0\n"
                "cmeq v6.16b, v6.16b, #0\n"
                "cmeq v7.16b, v7.16b, #0\n"
                "add v0.16b, v0.16b, v18.16b\n" // add palette offset
                "orr v2.16b, v2.16b, v6.16b\n" // combine tile and window visibility
                "add v1.16b, v1.16b, v19.16b\n"
                "orr v3.16b, v3.16b, v7.16b\n"

                weavePixels("%[paletteIndexPrimary]", "%[paletteIndexPrimary]")

                "add %[dst], %[dst], #128\n"
                "cmp %w[xoff], %w[xofftarget]\n"
                "blo loopStart%=\n"

            // do the remaining part
            "cmp %w[xoff], %w[xofftarget]\n"
            "sub %w[xoff], %w[xoff], #56\n"
            "sub %[dst], %[dst], #56*4\n"
            "sub %[windowMask], %[windowMask], #56\n"
            "bne loopStart%=\n"

            :
                asmDefaultOutput
            :
                asmDefaultInput(2), [paletteIndexPrimary] "w" (vdupq_n_u8(Num ? 2 : 0))
            :
                asmDefaultScratch
        );
    }
#undef loadTile
#undef loadPixels4
#undef loadPixels8
#undef loadMisc
#undef weavePixels
#undef asmDefaultOutput
#undef asmDefaultInput
#undef asmDefaultScratch
#undef regExtpalUse
}

#define DoDrawSprite(type, ...) \
    if (iswin) \
    { \
        DrawSprite_##type<true>(__VA_ARGS__); \
    } \
    else \
    { \
        DrawSprite_##type<false>(__VA_ARGS__); \
    }
void GPU2DNeon::DrawSprites(u32 line)
{
    /*if (line == 0)
    {
        // reset those counters here
        // TODO: find out when those are supposed to be reset
        // it would make sense to reset them at the end of VBlank
        // however, sprites are rendered one scanline in advance
        // so they need to be reset a bit earlier

        OBJMosaicY = 0;
        OBJMosaicYCount = 0;
    }

    NumSprites[0] = NumSprites[1] = NumSprites[2] = NumSprites[3] = 0;
    memset(OBJLine, 0, (256 + 8*2)*4);
    memset(OBJWindow, 0, 256 + 8*2);
    if (!(DispCnt & 0x1000)) return;

    memset(OBJIndex, 0xFF, 256 + 8*2);

    u16* oam = (u16*)&GPU::OAM[Num ? 0x400 : 0];

    const s32 spritewidth[16] =
    {
        8, 16, 8, 8,
        16, 32, 8, 8,
        32, 32, 16, 8,
        64, 64, 32, 8
    };
    const s32 spriteheight[16] =
    {
        8, 8, 16, 8,
        16, 8, 32, 8,
        32, 16, 32, 8,
        64, 32, 64, 8
    };

    for (int bgnum = 0x0C00; bgnum >= 0x0000; bgnum -= 0x0400)
    {
        for (int sprnum = 127; sprnum >= 0; sprnum--)
        {
            u16* attrib = &oam[sprnum*4];

            if ((attrib[2] & 0x0C00) != bgnum)
                continue;

            bool iswin = (((attrib[0] >> 10) & 0x3) == 2);

            u32 sprline;
            if ((attrib[0] & 0x1000) && !iswin)
            {
                // apply Y mosaic
                sprline = OBJMosaicY;
            }
            else
                sprline = line;

            if (attrib[0] & 0x0100)
            {
                u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
                s32 width = spritewidth[sizeparam];
                s32 height = spriteheight[sizeparam];
                s32 boundwidth = width;
                s32 boundheight = height;

                if (attrib[0] & 0x0200)
                {
                    boundwidth <<= 1;
                    boundheight <<= 1;
                }

                u32 ypos = attrib[0] & 0xFF;
                ypos = (sprline - ypos) & 0xFF;
                if (ypos >= (u32)boundheight)
                    continue;

                s32 xpos = (s32)(attrib[1] << 23) >> 23;
                if (xpos <= -boundwidth)
                    continue;

                u32 rotparamgroup = (attrib[1] >> 9) & 0x1F;

                //DoDrawSprite(Rotscale, sprnum, boundwidth, boundheight, width, height, xpos, ypos);

                NumSprites[(bgnum - 0xC00) / 0x400]++;
            }
            else
            {
                if (attrib[0] & 0x0200)
                    continue;

                u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
                s32 width = spritewidth[sizeparam];
                s32 height = spriteheight[sizeparam];

                u32 ypos = attrib[0] & 0xFF;
                ypos = (sprline - ypos) & 0xFF;
                if (ypos >= (u32)height)
                    continue;

                s32 xpos = (s32)(attrib[1] << 23) >> 23;
                if (xpos <= -width)
                    continue;

                DoDrawSprite(Normal, sprnum, width, height, xpos, ypos);

                NumSprites[(bgnum - 0xC00) / 0x400]++;
            }
        }
    }*/
}

/*
    Neue Idee!!!!!!!!
        alles wird erst am Ende in einem Schritt palettiert
*/

template<bool window>
void GPU2DNeon::DrawSprite_Normal(u32 num, u32 width, u32 height, s32 xpos, s32 ypos)
{
    u16* oam = (u16*)&GPU::OAM[Num ? 0x400 : 0];
    u16* attrib = &oam[num * 4];

    u32 pixelattr = ((attrib[2] & 0x0C00) << 6) | 0xC0000;
    u32 tilenum = attrib[2] & 0x03FF;
    u32 spritemode = window ? 0 : ((attrib[0] >> 10) & 0x3);

    u32 wmask = width - 8; // really ((width - 1) & ~0x7)

    if ((attrib[0] & 0x1000) && !window)
    {
        // apply Y mosaic
        pixelattr |= 0x100000;
    }

    // yflip
    if (attrib[1] & 0x2000)
        ypos = height-1 - ypos;

    u32 xoff;
    u32 xend = width;
    if (xpos >= 0)
    {
        xoff = 0;
        if ((xpos+xend) > 256)
            xend = 256 - xpos + (xoff & 0x7);
    }
    else
    {
        xoff = -(xpos & 0x7) - xpos;
        xpos = -(xpos & 0x7);
    }
    xpos += 8;

    if (spritemode == 3)
    {
        // bitmap sprite

        u16 color = 0; // transparent in all cases

        u32 alpha = attrib[2] >> 12;
        if (!alpha) return;
        alpha++;

        pixelattr |= (0xC0000000 | (alpha << 24));

        if (DispCnt & 0x40)
        {
            if (DispCnt & 0x20)
            {
                // 'reserved'
                // draws nothing

                return;
            }
            else
            {
                tilenum <<= (7 + ((DispCnt >> 22) & 0x1));
                tilenum += (ypos * width * 2);
            }
        }
        else
        {
            if (DispCnt & 0x20)
            {
                tilenum = ((tilenum & 0x01F) << 4) + ((tilenum & 0x3E0) << 7);
                tilenum += (ypos * 256 * 2);
            }
            else
            {
                tilenum = ((tilenum & 0x00F) << 4) + ((tilenum & 0x3F0) << 7);
                tilenum += (ypos * 128 * 2);
            }
        }

        u32 pixelsaddr = (Num ? 0x06600000 : 0x06400000) + tilenum;
        s32 pixelstride;

        if (attrib[1] & 0x1000) // xflip
        {
            pixelsaddr += (width-1 << 1);
            pixelsaddr -= (xoff << 1);
            pixelstride = -2;
        }
        else
        {
            pixelsaddr += (xoff << 1);
            pixelstride = 2;
        }

        /*for (; xoff < xend;)
        {
            color = GPU::ReadVRAM_OBJ<u16>(pixelsaddr);

            pixelsaddr += pixelstride;

            if (color & 0x8000)
            {
                if (window) OBJWindow[xpos] = 1;
                else      { OBJLine[xpos] = color | pixelattr; OBJIndex[xpos] = num; }
            }
            else if (!window)
            {
                if (OBJLine[xpos] == 0)
                {
                    OBJLine[xpos] = pixelattr & 0x180000;
                    OBJIndex[xpos] = num;
                }
            }

            xoff++;
            xpos++;
        }*/
    }
    else
    {
        if (DispCnt & 0x10)
        {
            tilenum <<= ((DispCnt >> 20) & 0x3);
            tilenum += ((ypos >> 3) * (width >> 3)) << ((attrib[0] & 0x2000) ? 1:0);
        }
        else
        {
            tilenum += ((ypos >> 3) * 0x20);
        }

        if (spritemode == 1) pixelattr |= 0x80000000;
        else                 pixelattr |= 0x10000000;

        if (attrib[0] & 0x2000)
        {
            // 256-color
            tilenum <<= 5;
            u32 pixelsaddr = (Num ? 0x06600000 : 0x06400000) + tilenum;
            pixelsaddr += ((ypos & 0x7) << 3);
            s32 pixelstride;

            if (!window)
            {
                if (!(DispCnt & 0x80000000))
                    pixelattr |= 0x1000;
                else
                    pixelattr |= ((attrib[2] & 0xF000) >> 4);
            }

            /*if (attrib[1] & 0x1000) // xflip
            {
                pixelsaddr += (((width-1) & wmask) << 3);
                pixelsaddr += ((width-1) & 0x7);
                pixelsaddr -= ((xoff & wmask) << 3);
                pixelsaddr -= (xoff & 0x7);
                pixelstride = -1;
            }
            else
            {*/
                pixelsaddr += ((xoff & wmask) << 3);
                pixelsaddr += (xoff & 0x7);
                pixelstride = 1;
            //}

            for (; xoff < xend; )
            {
                u64 pixels = GPU::ReadVRAM_OBJ<u64>(pixelsaddr);

                asm volatile (
                    "dup v0.2d, %[pixels]\n"
                    "\n"
                    :
                    :
                        [pixels] "r" (pixels)
                    :
                        "q0", "q1", "q2"
                );

                pixelsaddr += 56;
                xoff += 8;
                xpos += 8;
            }

            /*for (; xoff < xend;)
            {
                color = GPU::ReadVRAM_OBJ<u8>(pixelsaddr);

                pixelsaddr += pixelstride;

                if (color)
                {
                    if (window) OBJWindow[xpos] = 1;
                    else      { OBJLine[xpos] = color | pixelattr; OBJIndex[xpos] = num; }
                }
                else if (!window)
                {
                    if (OBJLine[xpos] == 0)
                    {
                        OBJLine[xpos] = pixelattr & 0x180000;
                        OBJIndex[xpos] = num;
                    }
                }

                xoff++;
                xpos++;
                if (!(xoff & 0x7)) pixelsaddr += (56 * pixelstride);
            }*/
        }
        else
        {
            // 16-color
            tilenum <<= 5;
            u32 pixelsaddr = (Num ? 0x06600000 : 0x06400000) + tilenum;
            pixelsaddr += ((ypos & 0x7) << 2);
            s32 pixelstride;

            if (!window)
            {
                pixelattr |= 0x1000;
                pixelattr |= ((attrib[2] & 0xF000) >> 8);
            }

            // TODO: optimize VRAM access!!
            // TODO: do xflip better? the 'two pixels per byte' thing makes it a bit shitty

            if (attrib[1] & 0x1000) // xflip
            {
                pixelsaddr += (((width-1) & wmask) << 2);
                pixelsaddr += (((width-1) & 0x7) >> 1);
                pixelsaddr -= ((xoff & wmask) << 2);
                pixelsaddr -= ((xoff & 0x7) >> 1);
                pixelstride = -1;
            }
            else
            {
                pixelsaddr += ((xoff & wmask) << 2);
                pixelsaddr += ((xoff & 0x7) >> 1);
                pixelstride = 1;
            }

            for (; xoff < xend & ~0x7;)
            {

            }
            /*for (; xoff < xend;)
            {
                if (attrib[1] & 0x1000)
                {
                    if (xoff & 0x1) { color = GPU::ReadVRAM_OBJ<u8>(pixelsaddr) & 0x0F; pixelsaddr--; }
                    else              color = GPU::ReadVRAM_OBJ<u8>(pixelsaddr) >> 4;
                }
                else
                {
                    if (xoff & 0x1) { color = GPU::ReadVRAM_OBJ<u8>(pixelsaddr) >> 4; pixelsaddr++; }
                    else              color = GPU::ReadVRAM_OBJ<u8>(pixelsaddr) & 0x0F;
                }

                if (color)
                {
                    if (window) OBJWindow[xpos] = 1;
                    else      { OBJLine[xpos] = color | pixelattr; OBJIndex[xpos] = num; }
                }
                else if (!window)
                {
                    if (OBJLine[xpos] == 0)
                    {
                        OBJLine[xpos] = pixelattr & 0x180000;
                        OBJIndex[xpos] = num;
                    }
                }

                xoff++;
                xpos++;
                if (!(xoff & 0x7)) pixelsaddr += ((attrib[1] & 0x1000) ? -28 : 28);
            }*/
        }
    }
}
