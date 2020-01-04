#include "GPU2D.h"

#include "NDS.h"
#include "GPU.h"

#include <string.h>
#include <arm_neon.h>

/*
    optimised GPU2D for less powerful, but NEON capable devices

    Q&A:
        Why inline assembler instead of intrinsics or dedicated assembler files?
            I would have loved to use intrinsics if GCC would produce good code for them.
            Sometimes it does, but once you cross the teritory of shrn/shrn2 and stuff 
            doing things like partial register writes the compiler output is just bad.
            Not even speaking of missing instructions such as ld1 with a set of four
            registers. 

            C++ assembler interop with separate files is just a no, so inline assembly
            is the only option left.
        Why are the NEON registers hard coded?
            Besides having full control over everything, instructions like ld1 or
            tbl require adjacent registers which the compiler can't really handle well.
*/

GPU2DNeon::GPU2DNeon(u32 num)
    : GPU2DBase(num)
{

}

void GPU2DNeon::SetDisplaySettings(bool accel)
{
    // OGL renderer is unsupported in conjunction with the Neon renderer
}

void GPU2DNeon::DrawScanline(u32 line)
{
    u32* dst = &Framebuffer[256 * line];

    int n3dline = line;
    line = GPU::VCount;
    
    if (Num == 0)
    {
        if ((CaptureCnt & (1<<31)) && (((CaptureCnt >> 29) & 0x3) != 1))
        {
            _3DLine = GPU3D::GetLine(n3dline);
            //GPU3D::GLRenderer::PrepareCaptureFrame();
        }
    }

    DrawScanline_BGOBJ(line);
    UpdateMosaicCounters(line);

    memcpy(dst, BGOBJLine, 256*4);
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
    u128 backdrop;
    if (Num) backdrop = *(u16*)&GPU::Palette[0x400];
    else     backdrop = *(u16*)&GPU::Palette[0];

    {
        u8 r = (backdrop & 0x001F) << 1;
        u8 g = (backdrop & 0x03E0) >> 4;
        u8 b = (backdrop & 0x7C00) >> 9;

        backdrop = r | (g << 8) | (b << 16) | 0x20000000;
        backdrop |= (backdrop << 32);
        backdrop |= (backdrop << 64);

        for (int i = 0; i < 256; i+=4)
            *(u128*)&BGOBJLine[i] = backdrop;
    }

    if (DispCnt & 0xE000)
        CalculateWindowMask(line);
    else
        memset(WindowMask, 0xFF, 256);

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
    u16* pal;
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
        tilesetaddr = (bgcnt & 0x003C) << 12;
        tilemapaddr = (bgcnt & 0x1F00) << 3;

        pal = (u16*)&GPU::Palette[0x400];
    }
    else
    {
        tilesetaddr = ((DispCnt & 0x07000000) >> 8) + ((bgcnt & 0x003C) << 12);
        tilemapaddr = ((DispCnt & 0x38000000) >> 11) + ((bgcnt & 0x1F00) << 3);

        pal = (u16*)&GPU::Palette[0];
    }
    
    // adjust Y position in tilemap
    if (bgcnt & 0x8000)
    {
        tilemapaddr += ((yoff & 0x1F8) << 3);
        if (bgcnt & 0x4000)
            tilemapaddr += ((yoff & 0x100) << 3);
    }
    else
        tilemapaddr += ((yoff & 0xF8) << 3);
    
    GPU::EnsureFlatVRAMCoherent(Num, tilemapaddr, 512 * 2);
    GPU::EnsureFlatVRAMCoherent(Num, tilesetaddr, 1024 * (bgcnt & 0x80 ? 64 : 32));

    u8* tilemapdata;
    u8* tilesetdata;
    if (Num)
    {
        tilemapdata = GPU::VRAMFlat_BBG + tilemapaddr;
        tilesetdata = GPU::VRAMFlat_BBG + tilesetaddr;
    }
    else
    {
        tilemapdata = GPU::VRAMFlat_ABG + tilemapaddr;
        tilesetdata = GPU::VRAMFlat_ABG + tilesetaddr;
    }

    uint8x16_t colorConvertMask = vdupq_n_u8(0x3E);
    uint8x16_t bgnumVec = vdupq_n_u8(bgnum);

    uint8x16_t scrollLUT;
    u32 amountSecondTile = (xoff & 0x7) ? (xoff & 0x7) : 8;
    u32 amountPrevTile = 8 - amountSecondTile;
    for (int i = 0; i < amountPrevTile; i++)
        scrollLUT[i] = 16 + 8 + amountSecondTile + i;
    for (int i = 0; i < 8 + amountSecondTile; i++)
        scrollLUT[amountPrevTile + i] = i;

    if (bgcnt & 0x0080)
    {
        for (int i = 0; i < 256; i += 16)
        {
            u16 curtile0 = *(u16*)(tilemapdata + ((xpos & 0xF8) >> 2) + ((xpos & widexmask) << 3));
            u16* curpal0;
            if (extpal) curpal0 = GetBGExtPal(extpalslot, curtile0>>12);
            else        curpal0 = pal;
            u64 pixels0 = *(tilesetdata + ((curtile0 & 0x03FF) << 6)
                                        + (((curtile0 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            xoff += 8;
            u16 curtile1 = *(u16*)(tilemapdata + ((xpos & 0xF8) >> 2) + ((xpos & widexmask) << 3));
            u16* curpal1;
            if (extpal) curpal1 = GetBGExtPal(extpalslot, curtile1>>12);
            else        curpal1 = pal;
            u64 pixels1 = *(tilesetdata + ((curtile1 & 0x03FF) << 6)
                                        + (((curtile1 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3));
            xoff += 8;

            if (curtile0 & 0x400)
                pixels0 = __builtin_bswap64(pixels0);
            if (curtile0 & 0x400)
                pixels1 = __builtin_bswap64(pixels1);

            u64 tmp, tmp2, tmp3, tmp4, tmp5, tmp6;
            asm volatile (
                "movi v8.16b, #0\n"
                "movi v10.16b, #0\n"
                "cbz %[pixels0], %=tile0Empty\n"

                // palettise
                "ubfx %[tmp], %[pixels0], #0, #8\n"
                "ldrh %w[tmp2], [%[curpal0], %[tmp]]\n"
                "ubfx %[tmp], %[pixels0], #0, #8\n"
                "ldrh %w[tmp3], [%[curpal0], %[tmp]]\n"
                "bfi %w[tmp2], %w[tmp3], #16, #16\n"
                "ins v8.4s[0], %w[tmp2]\n"

            "%=tile0Empty:\n"
                "cbz %[pixels1], %=tile1Empty\n"

            "%=tile1Empty:\n"
                :
                    [tmp] "=r" (tmp), [tmp2] "=r" (tmp2), [tmp3] "=r" (tmp3), [tmp4] "=r" (tmp4), [tmp6] "=r" (tmp6)
                :
                    [pixels0] "r" (pixels0), [pixels1] "r" (pixels1),
                    [curpal0] "r" (curpal0), [curpal1] "r" (curpal1)
                :
                    "memory"
            );
        }
        // 256-color

        // preload shit as needed
        /*if ((xoff & 0x7) || mosaic)
        {
            curtile = GPU::ReadVRAM_BG<u16>(tilemapaddr + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));

            if (extpal) curpal = GetBGExtPal(extpalslot, curtile>>12);
            else        curpal = pal;

            pixelsaddr = tilesetaddr + ((curtile & 0x03FF) << 6)
                                     + (((curtile & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3);
        }

        if (mosaic) lastxpos = xoff;

        for (int i = 0; i < 256; i++)
        {
            u32 xpos;
            if (mosaic) xpos = xoff - CurBGXMosaicTable[i];
            else        xpos = xoff;

            if ((!mosaic && (!(xpos & 0x7))) ||
                (mosaic && ((xpos >> 3) != (lastxpos >> 3))))
            {
                // load a new tile
                curtile = GPU::ReadVRAM_BG<u16>(tilemapaddr + ((xpos & 0xF8) >> 2) + ((xpos & widexmask) << 3));

                if (extpal) curpal = GetBGExtPal(extpalslot, curtile>>12);
                else        curpal = pal;

                pixelsaddr = tilesetaddr + ((curtile & 0x03FF) << 6)
                                         + (((curtile & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 3);

                if (mosaic) lastxpos = xpos;
            }

            // draw pixel
            if (WindowMask[i] & (1<<bgnum))
            {
                u32 tilexoff = (curtile & 0x0400) ? (7-(xpos&0x7)) : (xpos&0x7);
                color = GPU::ReadVRAM_BG<u8>(pixelsaddr + tilexoff);

                if (color)
                    DrawPixel(&BGOBJLine[i], curpal[color], 0x01000000<<bgnum);
            }

            xoff++;
        }*/
    }
    else
    {
        // 16-color

        // we need 4 adjacent registers for tbl lookup
        // second register is always the previous

        uint8x16_t indexOffset = {16, 16, 16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0};

        if (xoff & 0x7)
        {
            u16 curtile = *(u16*)(tilemapdata + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            u16* curpal = pal + ((curtile & 0xF000) >> 8);
            u32 pixels = *(u32*)(tilesetdata + ((curtile & 0x03FF) << 5)
                                        + (((curtile & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            if (curtile & 0x400)
            {
                pixels = __builtin_bswap32(pixels);
                pixels = ((pixels & 0xF0F0F0F0) >> 4) | ((pixels & 0x0F0F0F0F) << 4);
            }
            asm volatile (
                "dup v2.4s, %w[pixels]\n"
                // load palettes
                "ld2 {v3.16b, v4.16b}, [%[curpal]]\n"
                // unpack indices
                "shl v1.8b, v2.8b, #4\n"
                "ushr v0.8b, v2.8b, #4\n"
                "ushr v1.8b, v1.8b, #4\n"
                "zip1 v1.16b, v1.16b, v0.16b\n"

                "cmeq v13.16b, v1.16b, #0\n"

                // palettise
                "tbl v9.16b, {v3.16b}, v1.16b\n"
                "tbl v11.16b, {v4.16b}, v1.16b\n"
                :
                :
                    [pixels] "r" (pixels), [curpal] "r" (curpal)
                :
                    "q0", "q1", "q2", "q3", "q4", "q9", "q11", "q13"
            );
            xoff += 8;
        }

        for (int i = 0; i < 256; i += 16)
        {
            u16 curtile0 = *(u16*)(tilemapdata + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            u16* curpal0 = pal + ((curtile0 & 0xF000) >> 8);
            u32 pixels0 = *(u32*)(tilesetdata + ((curtile0 & 0x03FF) << 5)
                                        + (((curtile0 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            xoff += 8;
            u16 curtile1 = *(u16*)(tilemapdata + ((xoff & 0xF8) >> 2) + ((xoff & widexmask) << 3));
            u16* curpal1 = pal + ((curtile1 & 0xF000) >> 8);
            u32 pixels1 = *(u32*)(tilesetdata + ((curtile1 & 0x03FF) << 5)
                                        + (((curtile1 & 0x0800) ? (7-(yoff&0x7)) : (yoff&0x7)) << 2));
            xoff += 8;

            if (curtile0 & 0x400)
            {
                pixels0 = __builtin_bswap32(pixels0);
                pixels0 = ((pixels0 & 0xF0F0F0F0) >> 4) | ((pixels0 & 0x0F0F0F0F) << 4);
            }
            if (curtile1 & 0x400)
            {
                pixels1 = __builtin_bswap32(pixels1);
                pixels1 = ((pixels1 & 0xF0F0F0F0) >> 4) | ((pixels1 & 0x0F0F0F0F) << 4);
            }
            u64 pixels = (u64)pixels0 | ((u64)pixels1 << 32);

            // we do something potentially dangerous here
            // q13, q9 and q11 are used to preserve state between loop iterations
            asm volatile (
                "dup v2.2d, %[pixels]\n"

                // load palettes
                "ld2 {v3.16b, v4.16b}, [%[curpal0]]\n"
                "ld2 {v5.16b, v6.16b}, [%[curpal1]]\n"

                "ld4 {v19.16b, v20.16b, v21.16b, v22.16b}, [%[dst]]\n"
                "ld4 {v23.16b, v24.16b, v25.16b, v26.16b}, [%[dstBelow]]\n"

                "ld1 {v27.16b}, [%[windowMask]]\n"

                // unpack indices
                "shl v1.8b, v2.8b, #4\n"
                "ushr v0.8b, v2.8b, #4\n"
                "ushr v1.8b, v1.8b, #4\n"
                "zip1 v1.16b, v1.16b, v0.16b\n"

                // move palette so it can be used for tbl
                "mov v2.16b, v5.16b\n"
                "mov v7.16b, v4.16b\n"

                // generate transparency mask
                "cmeq v12.16b, v1.16b, #0\n"
                // for optimal scheduling the winodw code is a bit spread out
                "and v27.16b, v27.16b, %[bgnum].16b\n"

                "add v1.16b, v1.16b, %[indexOffset].16b\n"

                // apply scrolling to mask
                "tbl v0.16b, {v12.16b, v13.16b}, %[scrollLUT].16b\n"

                "cmeq v27.16b, v27.16b, #0\n"

                // palettise
                "tbl v8.16b, {v2.16b, v3.16b}, v1.16b\n"
                "tbl v10.16b, {v6.16b, v7.16b}, v1.16b\n"

                "orr v0.16b, v0.16b, v27.16b\n"

                "mov v13.16b, v12.16b\n"

                // move overriden values into second blending layer
                "bif v23.16b, v19.16b, v0.16b\n"
                "bif v24.16b, v20.16b, v0.16b\n"
                "bif v25.16b, v21.16b, v0.16b\n"
                "bif v26.16b, v22.16b, v0.16b\n"

                "st4 {v23.16b, v24.16b, v25.16b, v26.16b}, [%[dstBelow]]\n"

                // apply scrolling to color
                "tbl v3.16b, {v8.16b, v9.16b}, %[scrollLUT].16b\n"
                "tbl v4.16b, {v10.16b, v11.16b}, %[scrollLUT].16b\n"

                "mov v9.16b, v8.16b\n"
                "mov v11.16b, v10.16b\n"

                // convert 5-bit to 6-bit colors
                "ushr v5.16b, v4.16b, #1\n"
                "zip2 v6.16b, v3.16b, v4.16b\n"
                "zip1 v4.16b, v3.16b, v4.16b\n"
                "shl v3.16b, v3.16b, #1\n"
                "shrn v4.8b, v4.8h, #4\n"
                "shrn2 v4.16b, v6.8h, #4\n"
                "and v3.16b, v3.16b, %[colorConvertMask].16b\n"
                "and v5.16b, v5.16b, %[colorConvertMask].16b\n"
                "and v4.16b, v4.16b, %[colorConvertMask].16b\n"

                // move new values into first blending layer
                "bif v19.16b, v3.16b, v0.16b\n"
                "bif v20.16b, v4.16b, v0.16b\n"
                "bif v21.16b, v5.16b, v0.16b\n"
                "bif v22.16b, %[bgnum].16b, v0.16b\n"

                "st4 {v19.16b, v20.16b, v21.16b, v22.16b}, [%[dst]]\n"
                :
                :
                    [curpal0] "r" (curpal0), [curpal1] "r" (curpal1),
                    [pixels] "r" (pixels),
                    [dst] "r" (&BGOBJLine[i]), [dstBelow] "r" (&BGOBJLine[i + 256]),
                    [indexOffset] "w" (indexOffset),
                    [colorConvertMask] "w" (colorConvertMask),
                    [bgnum] "w" (bgnumVec),
                    [scrollLUT] "w" (scrollLUT),
                    [windowMask] "r" (&WindowMask[i])
                :
                    "memory",
                    "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                    "q8", "q9", "q10", "q11", "q12", "q13", "q19", 
                    "q20", "q21", "q22", "q23", "q24", "q25", "q26", "q27"
            );
        }
    }
}

void GPU2DNeon::DrawSprites(u32 line)
{

}