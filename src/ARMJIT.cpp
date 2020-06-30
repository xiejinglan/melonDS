#include "ARMJIT.h"

#include <string.h>
#include <assert.h>
#include <unordered_map>

#define XXH_STATIC_LINKING_ONLY
#include "xxhash/xxhash.h"

#include "Config.h"

#include "ARMJIT_Internal.h"
#include "ARMJIT_Memory.h"
#include "ARMJIT_Compiler.h"

#include "ARMInterpreter_ALU.h"
#include "ARMInterpreter_LoadStore.h"
#include "ARMInterpreter_Branch.h"
#include "ARMInterpreter.h"

#include "DSi.h"
#include "GPU.h"
#include "GPU3D.h"
#include "SPU.h"
#include "Wifi.h"
#include "NDSCart.h"

#include "ARMJIT_x64/ARMJIT_Offsets.h"
static_assert(offsetof(ARM, CPSR) == ARM_CPSR_offset);
static_assert(offsetof(ARM, Cycles) == ARM_Cycles_offset);
static_assert(offsetof(ARM, StopExecution) == ARM_StopExecution_offset);

namespace ARMJIT
{

#define JIT_DEBUGPRINT(msg, ...)
//#define JIT_DEBUGPRINT(msg, ...) printf(msg, ## __VA_ARGS__)

Compiler* JITCompiler;

AddressRange CodeIndexITCM[ITCMPhysicalSize / 512];
AddressRange CodeIndexMainRAM[NDS::MainRAMMaxSize / 512];
AddressRange CodeIndexSWRAM[NDS::SharedWRAMSize / 512];
AddressRange CodeIndexVRAM[0x100000 / 512];
AddressRange CodeIndexARM9BIOS[sizeof(NDS::ARM9BIOS) / 512];
AddressRange CodeIndexARM7BIOS[sizeof(NDS::ARM7BIOS) / 512];
AddressRange CodeIndexARM7WRAM[NDS::ARM7WRAMSize / 512];
AddressRange CodeIndexARM7WVRAM[0x40000 / 512];
AddressRange CodeIndexBIOS9DSi[0x10000 / 512];
AddressRange CodeIndexBIOS7DSi[0x10000 / 512];
AddressRange CodeIndexNWRAM_A[DSi::NWRAMSize / 512];
AddressRange CodeIndexNWRAM_B[DSi::NWRAMSize / 512];
AddressRange CodeIndexNWRAM_C[DSi::NWRAMSize / 512];

std::unordered_map<u32, JitBlock*> JitBlocks9;
std::unordered_map<u32, JitBlock*> JitBlocks7;

u64 FastBlockLookupITCM[ITCMPhysicalSize / 2];
u64 FastBlockLookupMainRAM[NDS::MainRAMMaxSize / 2];
u64 FastBlockLookupSWRAM[NDS::SharedWRAMSize / 2];
u64 FastBlockLookupVRAM[0x100000 / 2];
u64 FastBlockLookupARM9BIOS[sizeof(NDS::ARM9BIOS) / 2];
u64 FastBlockLookupARM7BIOS[sizeof(NDS::ARM7BIOS) / 2];
u64 FastBlockLookupARM7WRAM[NDS::ARM7WRAMSize / 2];
u64 FastBlockLookupARM7WVRAM[0x40000 / 2];
u64 FastBlockLookupBIOS9DSi[0x10000 / 2];
u64 FastBlockLookupBIOS7DSi[0x10000 / 2];
u64 FastBlockLookupNWRAM_A[DSi::NWRAMSize / 2];
u64 FastBlockLookupNWRAM_B[DSi::NWRAMSize / 2];
u64 FastBlockLookupNWRAM_C[DSi::NWRAMSize / 2];

const u32 CodeRegionSizes[ARMJIT_Memory::memregions_Count] =
{
	0,
	ITCMPhysicalSize,
	0,
	sizeof(NDS::ARM9BIOS),
	NDS::MainRAMMaxSize,
	NDS::SharedWRAMSize,
	0,
	0x100000,
	sizeof(NDS::ARM7BIOS),
	NDS::ARM7WRAMSize,
	0,
	0,
	0x40000,
	0x10000,
	0x10000,
	sizeof(DSi::NWRAM_A),
	sizeof(DSi::NWRAM_B),
	sizeof(DSi::NWRAM_C),
};

AddressRange* const CodeMemRegions[ARMJIT_Memory::memregions_Count] =
{
	NULL,
	CodeIndexITCM,
	NULL,
	CodeIndexARM9BIOS,
	CodeIndexMainRAM,
	CodeIndexSWRAM,
	NULL,
	CodeIndexVRAM,
	CodeIndexARM7BIOS,
	CodeIndexARM7WRAM,
	NULL,
	NULL,
	CodeIndexARM7WVRAM,
	CodeIndexBIOS9DSi,
	CodeIndexBIOS7DSi,
	CodeIndexNWRAM_A,
	CodeIndexNWRAM_B,
	CodeIndexNWRAM_C
};

u64* const FastBlockLookupRegions[ARMJIT_Memory::memregions_Count] =
{
	NULL,
	FastBlockLookupITCM,
	NULL,
	FastBlockLookupARM9BIOS,
	FastBlockLookupMainRAM,
	FastBlockLookupSWRAM,
	NULL,
	FastBlockLookupVRAM,
	FastBlockLookupARM7BIOS,
	FastBlockLookupARM7WRAM,
	NULL,
	NULL,
	FastBlockLookupARM7WVRAM,
	FastBlockLookupBIOS9DSi,
	FastBlockLookupBIOS7DSi,
	FastBlockLookupNWRAM_A,
	FastBlockLookupNWRAM_B,
	FastBlockLookupNWRAM_C
};

u32 LocaliseCodeAddress(u32 num, u32 addr)
{
	int region = num == 0
		? ARMJIT_Memory::ClassifyAddress9(addr)
		: ARMJIT_Memory::ClassifyAddress7(addr);

	if (CodeMemRegions[region])
		return ARMJIT_Memory::LocaliseAddress(region, num, addr);
	return 0;
}

TinyVector<u32> InvalidLiterals;

template <typename T, int ConsoleType>
T SlowRead9(u32 addr, ARMv5* cpu)
{
	u32 offset = addr & 0x3;
	addr &= ~(sizeof(T) - 1);

	T val;
	if (addr < cpu->ITCMSize)
		val = *(T*)&cpu->ITCM[addr & 0x7FFF];
	else if (addr >= cpu->DTCMBase && addr < (cpu->DTCMBase + cpu->DTCMSize))
		val = *(T*)&cpu->DTCM[(addr - cpu->DTCMBase) & 0x3FFF];
	else if (std::is_same<T, u32>::value)
		val = (ConsoleType == 0 ? NDS::ARM9Read32 : DSi::ARM9Read32)(addr);
	else if (std::is_same<T, u16>::value)
		val = (ConsoleType == 0 ? NDS::ARM9Read16 : DSi::ARM9Read16)(addr);
	else
		val = (ConsoleType == 0 ? NDS::ARM9Read8 : DSi::ARM9Read8)(addr);

	if (std::is_same<T, u32>::value)
		return ROR(val, offset << 3);
	else
		return val;
}

template <typename T, int ConsoleType>
void SlowWrite9(u32 addr, ARMv5* cpu, T val)
{
	addr &= ~(sizeof(T) - 1);

    if (addr < cpu->ITCMSize)
	{
        CheckAndInvalidate<0, ARMJIT_Memory::memregion_ITCM>(addr);
		*(T*)&cpu->ITCM[addr & 0x7FFF] = val;
	}
	else if (addr >= cpu->DTCMBase && addr < (cpu->DTCMBase + cpu->DTCMSize))
	{
		*(T*)&cpu->DTCM[(addr - cpu->DTCMBase) & 0x3FFF] = val;
	}
	else if (std::is_same<T, u32>::value)
	{
		(ConsoleType == 0 ? NDS::ARM9Write32 : DSi::ARM9Write32)(addr, val);
	}
	else if (std::is_same<T, u16>::value)
	{
		(ConsoleType == 0 ? NDS::ARM9Write16 : DSi::ARM9Write16)(addr, val);
	}
	else
	{
		(ConsoleType == 0 ? NDS::ARM9Write8 : DSi::ARM9Write8)(addr, val);
	}
}

template <typename T, int ConsoleType>
T SlowRead7(u32 addr)
{
	u32 offset = addr & 0x3;
	addr &= ~(sizeof(T) - 1);

	T val;
	if (std::is_same<T, u32>::value)
		val = (ConsoleType == 0 ? NDS::ARM7Read32 : DSi::ARM7Read32)(addr);
	else if (std::is_same<T, u16>::value)
		val = (ConsoleType == 0 ? NDS::ARM7Read16 : DSi::ARM7Read16)(addr);
	else
		val = (ConsoleType == 0 ? NDS::ARM7Read8 : DSi::ARM7Read8)(addr);

	if (std::is_same<T, u32>::value)
		return ROR(val, offset << 3);
	else
		return val;
}

template <typename T, int ConsoleType>
void SlowWrite7(u32 addr, T val)
{
	addr &= ~(sizeof(T) - 1);

	if (std::is_same<T, u32>::value)
		(ConsoleType == 0 ? NDS::ARM7Write32 : DSi::ARM7Write32)(addr, val);
	else if (std::is_same<T, u16>::value)
		(ConsoleType == 0 ? NDS::ARM7Write16 : DSi::ARM7Write16)(addr, val);
	else
		(ConsoleType == 0 ? NDS::ARM7Write8 : DSi::ARM7Write8)(addr, val);
}

template <bool Write, int ConsoleType>
void SlowBlockTransfer9(u32 addr, u64* data, u32 num, ARMv5* cpu)
{
	addr &= ~0x3;
	for (int i = 0; i < num; i++)
	{
		if (Write)
			SlowWrite9<u32, ConsoleType>(addr, cpu, data[i]);
		else
			data[i] = SlowRead9<u32, ConsoleType>(addr, cpu);
		addr += 4;
	}
}

template <bool Write, int ConsoleType>
void SlowBlockTransfer7(u32 addr, u64* data, u32 num)
{
	addr &= ~0x3;
	for (int i = 0; i < num; i++)
	{
		if (Write)
			SlowWrite7<u32, ConsoleType>(addr, data[i]);
		else
			data[i] = SlowRead7<u32, ConsoleType>(addr);
		addr += 4;
	}
}

#define INSTANTIATE_SLOWMEM(consoleType) \
	template void SlowWrite9<u32, consoleType>(u32, ARMv5*, u32); \
	template void SlowWrite9<u16, consoleType>(u32, ARMv5*, u16); \
	template void SlowWrite9<u8, consoleType>(u32, ARMv5*, u8); \
	\
	template u32 SlowRead9<u32, consoleType>(u32, ARMv5*); \
	template u16 SlowRead9<u16, consoleType>(u32, ARMv5*); \
	template u8 SlowRead9<u8, consoleType>(u32, ARMv5*); \
	\
	template void SlowWrite7<u32, consoleType>(u32, u32); \
	template void SlowWrite7<u16, consoleType>(u32, u16); \
	template void SlowWrite7<u8, consoleType>(u32, u8); \
	\
	template u32 SlowRead7<u32, consoleType>(u32); \
	template u16 SlowRead7<u16, consoleType>(u32); \
	template u8 SlowRead7<u8, consoleType>(u32); \
	\
	template void SlowBlockTransfer9<false, consoleType>(u32, u64*, u32, ARMv5*); \
	template void SlowBlockTransfer9<true, consoleType>(u32, u64*, u32, ARMv5*); \
	template void SlowBlockTransfer7<false, consoleType>(u32 addr, u64* data, u32 num); \
	template void SlowBlockTransfer7<true, consoleType>(u32 addr, u64* data, u32 num); \

INSTANTIATE_SLOWMEM(0)
INSTANTIATE_SLOWMEM(1)

template <typename K, typename V, int Size, V InvalidValue>
struct UnreliableHashTable
{
	struct Bucket
	{
		K KeyA, KeyB;
		V ValA, ValB;
	};

	Bucket Table[Size];

	void Reset()
	{
		for (int i = 0; i < Size; i++)
		{
			Table[i].ValA = Table[i].ValB = InvalidValue;
		}
	}

	UnreliableHashTable()
	{
		Reset();
	}

	V Insert(K key, V value)
	{
		u32 slot = XXH3_64bits(&key, sizeof(K)) & (Size - 1);
		Bucket* bucket = &Table[slot];

		if (bucket->ValA == value || bucket->ValB == value)
		{
			return InvalidValue;
		}
		else if (bucket->ValA == InvalidValue)
		{
			bucket->KeyA = key;
			bucket->ValA = value;
		}
		else if (bucket->ValB == InvalidValue)
		{
			bucket->KeyB = key;
			bucket->ValB = value;
		}
		else
		{
			V prevVal = bucket->ValB;
			bucket->KeyB = bucket->KeyA;
			bucket->ValB = bucket->ValA;
			bucket->KeyA = key;
			bucket->ValA = value;
			return prevVal;
		}

		return InvalidValue;
	}

	void Remove(K key)
	{
		u32 slot = XXH3_64bits(&key, sizeof(K)) & (Size - 1);
		Bucket* bucket = &Table[slot];

		if (bucket->KeyA == key && bucket->ValA != InvalidValue)
		{
			bucket->ValA = InvalidValue;
			if (bucket->ValB != InvalidValue)
			{
				bucket->KeyA = bucket->KeyB;
				bucket->ValA = bucket->ValB;
				bucket->ValB = InvalidValue;
			}
		}
		if (bucket->KeyB == key && bucket->ValB != InvalidValue)
			bucket->ValB = InvalidValue;
	}

	V LookUp(K addr)
	{
		u32 slot = XXH3_64bits(&addr, 4) & (Size - 1);
		Bucket* bucket = &Table[slot];

		if (bucket->ValA != InvalidValue && bucket->KeyA == addr)
			return bucket->ValA;
		if (bucket->ValB != InvalidValue && bucket->KeyB == addr)
			return bucket->ValB;

		return InvalidValue;
	}
};

UnreliableHashTable<u32, JitBlock*, 0x800, nullptr> RestoreCandidates;

void Init()
{
	JITCompiler = new Compiler();

	ARMJIT_Memory::Init();
}

void DeInit()
{
	ARMJIT_Memory::DeInit();

	delete JITCompiler;
}

void Reset()
{
	ResetBlockCache();

	ARMJIT_Memory::Reset();
}

void FloodFillSetFlags(FetchedInstr instrs[], int start, u8 flags)
{
	for (int j = start; j >= 0; j--)
	{
		u8 match = instrs[j].Info.WriteFlags & flags;
		u8 matchMaybe = (instrs[j].Info.WriteFlags >> 4) & flags;
		if (matchMaybe) // writes flags maybe
			instrs[j].SetFlags |= matchMaybe;
		if (match)
		{
			instrs[j].SetFlags |= match;
			flags &= ~match;
			if (!flags)
				return;
		}
	}
}

bool DecodeLiteral(bool thumb, const FetchedInstr& instr, u32& addr)
{
	if (!thumb)
	{
		switch (instr.Info.Kind)
		{
		case ARMInstrInfo::ak_LDR_IMM:
		case ARMInstrInfo::ak_LDRB_IMM:
			addr = (instr.Addr + 8) + ((instr.Instr & 0xFFF) * (instr.Instr & (1 << 23) ? 1 : -1));
			return true;
		case ARMInstrInfo::ak_LDRH_IMM:
			addr = (instr.Addr + 8) + (((instr.Instr & 0xF00) >> 4 | (instr.Instr & 0xF)) * (instr.Instr & (1 << 23) ? 1 : -1));
			return true;
		default:
			break;
		}
	}
	else if (instr.Info.Kind == ARMInstrInfo::tk_LDR_PCREL)
	{
    	addr = ((instr.Addr + 4) & ~0x2) + ((instr.Instr & 0xFF) << 2);
		return true;
	}

	JIT_DEBUGPRINT("Literal %08x %x not recognised %d\n", instr.Instr, instr.Addr, instr.Info.Kind);
	return false;
}

bool DecodeBranch(bool thumb, const FetchedInstr& instr, u32& cond, bool hasLink, u32 lr, bool& link, 
	u32& linkAddr, u32& targetAddr)
{
	if (thumb)
	{
		u32 r15 = instr.Addr + 4;
		cond = 0xE;

		link = instr.Info.Kind == ARMInstrInfo::tk_BL_LONG;
		linkAddr = instr.Addr + 4;

		if (instr.Info.Kind == ARMInstrInfo::tk_BL_LONG && !(instr.Instr & (1 << 12)))
		{
			targetAddr = r15 + ((s32)((instr.Instr & 0x7FF) << 21) >> 9);
    		targetAddr += ((instr.Instr >> 16) & 0x7FF) << 1;
			return true;
		}
		else if (instr.Info.Kind == ARMInstrInfo::tk_B)
		{
			s32 offset = (s32)((instr.Instr & 0x7FF) << 21) >> 20;
			targetAddr = r15 + offset;
			return true;
		}
		else if (instr.Info.Kind == ARMInstrInfo::tk_BCOND)
		{
			cond = (instr.Instr >> 8) & 0xF;
			s32 offset = (s32)(instr.Instr << 24) >> 23;
			targetAddr = r15 + offset;
			return true;
		}
		else if (hasLink && instr.Info.Kind == ARMInstrInfo::tk_BX && instr.A_Reg(3) == 14)
		{
			JIT_DEBUGPRINT("returning!\n");
			targetAddr = lr;
			return true;
		}
	}
	else
	{
		link = instr.Info.Kind == ARMInstrInfo::ak_BL;
		linkAddr = instr.Addr + 4;

		cond = instr.Cond();
		if (instr.Info.Kind == ARMInstrInfo::ak_BL 
			|| instr.Info.Kind == ARMInstrInfo::ak_B)
		{
			s32 offset = (s32)(instr.Instr << 8) >> 6;
			u32 r15 = instr.Addr + 8;
			targetAddr = r15 + offset;
			return true;
		}
		else if (hasLink && instr.Info.Kind == ARMInstrInfo::ak_BX && instr.A_Reg(0) == 14)
		{
			JIT_DEBUGPRINT("returning!\n");
			targetAddr = lr;
			return true;
		}
	}
	return false;
}

bool IsIdleLoop(FetchedInstr* instrs, int instrsCount)
{
	// see https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/Core/PowerPC/PPCAnalyst.cpp#L678
	// it basically checks if one iteration of a loop depends on another
	// the rules are quite simple

	JIT_DEBUGPRINT("checking potential idle loop\n");
	u16 regsWrittenTo = 0;
	u16 regsDisallowedToWrite = 0;
	for (int i = 0; i < instrsCount; i++)
	{
		JIT_DEBUGPRINT("instr %d %x regs(%x %x) %x %x\n", i, instrs[i].Instr, instrs[i].Info.DstRegs, instrs[i].Info.SrcRegs, regsWrittenTo, regsDisallowedToWrite);
		if (instrs[i].Info.SpecialKind == ARMInstrInfo::special_WriteMem)
			return false;
		if (i < instrsCount - 1 && instrs[i].Info.Branches())
			return false;

		u16 srcRegs = instrs[i].Info.SrcRegs & ~(1 << 15);
		u16 dstRegs = instrs[i].Info.DstRegs & ~(1 << 15);

		regsDisallowedToWrite |= srcRegs & ~regsWrittenTo;
		
		if (dstRegs & regsDisallowedToWrite)
			return false;
		regsWrittenTo |= dstRegs;
	}
	return true;
}

typedef void (*InterpreterFunc)(ARM* cpu);

void NOP(ARM* cpu) {}

#define F(x) &ARMInterpreter::A_##x
#define F_ALU(name, s) \
	F(name##_REG_LSL_IMM##s), F(name##_REG_LSR_IMM##s), F(name##_REG_ASR_IMM##s), F(name##_REG_ROR_IMM##s), \
	F(name##_REG_LSL_REG##s), F(name##_REG_LSR_REG##s), F(name##_REG_ASR_REG##s), F(name##_REG_ROR_REG##s), F(name##_IMM##s)
#define F_MEM_WB(name) \
	F(name##_REG_LSL), F(name##_REG_LSR), F(name##_REG_ASR), F(name##_REG_ROR), F(name##_IMM), \
	F(name##_POST_REG_LSL), F(name##_POST_REG_LSR), F(name##_POST_REG_ASR), F(name##_POST_REG_ROR), F(name##_POST_IMM)
#define F_MEM_HD(name) \
	F(name##_REG), F(name##_IMM), F(name##_POST_REG), F(name##_POST_IMM)
InterpreterFunc InterpretARM[ARMInstrInfo::ak_Count] =
{
	F_ALU(AND,), F_ALU(AND,_S),
	F_ALU(EOR,), F_ALU(EOR,_S),
	F_ALU(SUB,), F_ALU(SUB,_S),
	F_ALU(RSB,), F_ALU(RSB,_S),
	F_ALU(ADD,), F_ALU(ADD,_S),
	F_ALU(ADC,), F_ALU(ADC,_S),
	F_ALU(SBC,), F_ALU(SBC,_S),
	F_ALU(RSC,), F_ALU(RSC,_S),
	F_ALU(ORR,), F_ALU(ORR,_S),
	F_ALU(MOV,), F_ALU(MOV,_S),
	F_ALU(BIC,), F_ALU(BIC,_S),
	F_ALU(MVN,), F_ALU(MVN,_S),
	F_ALU(TST,),
	F_ALU(TEQ,),
	F_ALU(CMP,),
	F_ALU(CMN,),

	F(MUL), F(MLA), F(UMULL), F(UMLAL), F(SMULL), F(SMLAL), F(SMLAxy), F(SMLAWy), F(SMULWy), F(SMLALxy), F(SMULxy),
	F(CLZ), F(QADD), F(QDADD), F(QSUB), F(QDSUB),

	F_MEM_WB(STR),
	F_MEM_WB(STRB),
	F_MEM_WB(LDR),
	F_MEM_WB(LDRB),

	F_MEM_HD(STRH),
	F_MEM_HD(LDRD),
	F_MEM_HD(STRD),
	F_MEM_HD(LDRH),
	F_MEM_HD(LDRSB),
	F_MEM_HD(LDRSH),

	F(SWP), F(SWPB),
	F(LDM), F(STM),

	F(B), F(BL), F(BLX_IMM), F(BX), F(BLX_REG),
	F(UNK), F(MSR_IMM), F(MSR_REG), F(MRS), F(MCR), F(MRC), F(SVC),
	NOP
};
#undef F_ALU
#undef F_MEM_WB
#undef F_MEM_HD
#undef F

void T_BL_LONG(ARM* cpu)
{
	ARMInterpreter::T_BL_LONG_1(cpu);
	cpu->R[15] += 2;
	ARMInterpreter::T_BL_LONG_2(cpu);
}

#define F(x) ARMInterpreter::T_##x
InterpreterFunc InterpretTHUMB[ARMInstrInfo::tk_Count] =
{
	F(LSL_IMM), F(LSR_IMM), F(ASR_IMM),
	F(ADD_REG_), F(SUB_REG_), F(ADD_IMM_), F(SUB_IMM_),
	F(MOV_IMM), F(CMP_IMM), F(ADD_IMM), F(SUB_IMM),
	F(AND_REG), F(EOR_REG), F(LSL_REG), F(LSR_REG), F(ASR_REG),
	F(ADC_REG), F(SBC_REG), F(ROR_REG), F(TST_REG), F(NEG_REG),
	F(CMP_REG), F(CMN_REG), F(ORR_REG), F(MUL_REG), F(BIC_REG), F(MVN_REG),
	F(ADD_HIREG), F(CMP_HIREG), F(MOV_HIREG),
	F(ADD_PCREL), F(ADD_SPREL), F(ADD_SP),
	F(LDR_PCREL), F(STR_REG), F(STRB_REG), F(LDR_REG), F(LDRB_REG), F(STRH_REG),
	F(LDRSB_REG), F(LDRH_REG), F(LDRSH_REG), F(STR_IMM), F(LDR_IMM), F(STRB_IMM),
	F(LDRB_IMM), F(STRH_IMM), F(LDRH_IMM), F(STR_SPREL), F(LDR_SPREL),
	F(PUSH), F(POP), F(LDMIA), F(STMIA),
	F(BCOND), F(BX), F(BLX_REG), F(B), F(BL_LONG_1), F(BL_LONG_2),
	F(UNK), F(SVC), 
	T_BL_LONG // BL_LONG psudo opcode
};
#undef F

void CompileBlock(ARM* cpu)
{
    bool thumb = cpu->CPSR & 0x20;

	if (Config::JIT_MaxBlockSize < 1)
		Config::JIT_MaxBlockSize = 1;
	if (Config::JIT_MaxBlockSize > 32)
		Config::JIT_MaxBlockSize = 32;

	u32 blockAddr = cpu->R[15] - (thumb ? 2 : 4);

	u32 localAddr = LocaliseCodeAddress(cpu->Num, blockAddr);
	if (!localAddr)
	{
		printf("trying to compile non executable code? %x\n", blockAddr);
	}

	auto& map = cpu->Num == 0 ? JitBlocks9 : JitBlocks7;
	auto existingBlockIt = map.find(blockAddr);
	if (existingBlockIt != map.end())
	{
		// there's already a block, though it's not inside the fast map
		// could be that there are two blocks at the same physical addr
		// but different mirrors
		u32 otherLocalAddr = existingBlockIt->second->StartAddrLocal;

		if (localAddr == otherLocalAddr)
		{
			JIT_DEBUGPRINT("switching out block %x %x %x\n", localAddr, blockAddr, existingBlockIt->second->StartAddr);

			u64* entry = &FastBlockLookupRegions[localAddr >> 27][(localAddr & 0x7FFFFFF) / 2];
			*entry = ((u64)blockAddr | cpu->Num) << 32;
			*entry |= JITCompiler->SubEntryOffset(existingBlockIt->second->EntryPoint);
			return;
		}

		// some memory has been remapped
		JitBlock* prevBlock = RestoreCandidates.Insert(existingBlockIt->second->InstrHash, existingBlockIt->second);
		if (prevBlock)
			delete prevBlock;
		
		map.erase(existingBlockIt);
	}

    FetchedInstr instrs[Config::JIT_MaxBlockSize];
    int i = 0;
    u32 r15 = cpu->R[15];

	u32 addressRanges[Config::JIT_MaxBlockSize];
	u32 addressMasks[Config::JIT_MaxBlockSize] = {0};
	u32 numAddressRanges = 0;

	u32 numLiterals = 0;
	u32 literalLoadAddrs[Config::JIT_MaxBlockSize];
	// they are going to be hashed
	u32 literalValues[Config::JIT_MaxBlockSize];
	u32 instrValues[Config::JIT_MaxBlockSize];

	cpu->FillPipeline();
    u32 nextInstr[2] = {cpu->NextInstr[0], cpu->NextInstr[1]};
	u32 nextInstrAddr[2] = {blockAddr, r15};

	JIT_DEBUGPRINT("start block %x %08x (%x)\n", blockAddr, cpu->CPSR, localAddr);

	u32 lastSegmentStart = blockAddr;
	u32 lr;
	bool hasLink = false;

    do
    {
        r15 += thumb ? 2 : 4;

		instrs[i].BranchFlags = 0;
		instrs[i].SetFlags = 0;
        instrs[i].Instr = nextInstr[0];
        nextInstr[0] = nextInstr[1];
	
		instrs[i].Addr = nextInstrAddr[0];
		nextInstrAddr[0] = nextInstrAddr[1];
		nextInstrAddr[1] = r15;
		JIT_DEBUGPRINT("instr %08x %x\n", instrs[i].Instr & (thumb ? 0xFFFF : ~0), instrs[i].Addr);

		instrValues[i] = instrs[i].Instr;

		u32 translatedAddr = LocaliseCodeAddress(cpu->Num, instrs[i].Addr);
		assert(translatedAddr >> 27);
		u32 translatedAddrRounded = translatedAddr & ~0x1FF;
		if (i == 0 || translatedAddrRounded != addressRanges[numAddressRanges - 1])
		{
			bool returning = false;
			for (int j = 0; j < numAddressRanges; j++)
			{
				if (addressRanges[j] == translatedAddrRounded)
				{
					std::swap(addressRanges[j], addressRanges[numAddressRanges - 1]);
					std::swap(addressMasks[j], addressMasks[numAddressRanges - 1]);
					returning = true;
					break;
				}
			}
			if (!returning)
				addressRanges[numAddressRanges++] = translatedAddrRounded;
		}
		addressMasks[numAddressRanges - 1] |= 1 << ((translatedAddr & 0x1FF) / 16);

        if (cpu->Num == 0)
        {
            ARMv5* cpuv5 = (ARMv5*)cpu;
            if (thumb && r15 & 0x2)
            {
                nextInstr[1] >>= 16;
                instrs[i].CodeCycles = 0;
            }
            else
            {
                nextInstr[1] = cpuv5->CodeRead32(r15, false);
                instrs[i].CodeCycles = cpu->CodeCycles;
            }
        }
        else
        {
            ARMv4* cpuv4 = (ARMv4*)cpu;
            if (thumb)
                nextInstr[1] = cpuv4->CodeRead16(r15);
            else
                nextInstr[1] = cpuv4->CodeRead32(r15);
            instrs[i].CodeCycles = cpu->CodeCycles;
        }
        instrs[i].Info = ARMInstrInfo::Decode(thumb, cpu->Num, instrs[i].Instr);

		cpu->R[15] = r15;
		cpu->CurInstr = instrs[i].Instr;
		cpu->CodeCycles = instrs[i].CodeCycles;

		if (instrs[i].Info.DstRegs & (1 << 14)
			|| (!thumb
				&& (instrs[i].Info.Kind == ARMInstrInfo::ak_MSR_IMM || instrs[i].Info.Kind == ARMInstrInfo::ak_MSR_REG)
				&& instrs[i].Instr & (1 << 16)))
			hasLink = false;

		if (thumb)
		{
			InterpretTHUMB[instrs[i].Info.Kind](cpu);
		}
		else
		{
			if (cpu->Num == 0 && instrs[i].Info.Kind == ARMInstrInfo::ak_BLX_IMM)
			{
				ARMInterpreter::A_BLX_IMM(cpu);
			}
			else
			{
                u32 icode = ((instrs[i].Instr >> 4) & 0xF) | ((instrs[i].Instr >> 16) & 0xFF0);
				assert(InterpretARM[instrs[i].Info.Kind] == ARMInterpreter::ARMInstrTable[icode]
					|| instrs[i].Info.Kind == ARMInstrInfo::ak_MOV_REG_LSL_IMM
					|| instrs[i].Info.Kind == ARMInstrInfo::ak_Nop
					|| instrs[i].Info.Kind == ARMInstrInfo::ak_UNK);
				if (cpu->CheckCondition(instrs[i].Cond()))
					InterpretARM[instrs[i].Info.Kind](cpu);
				else
					cpu->AddCycles_C();
			}
		}

		instrs[i].DataCycles = cpu->DataCycles;
		instrs[i].DataRegion = cpu->DataRegion;

		u32 literalAddr;
		if (Config::JIT_LiteralOptimisations
			&& instrs[i].Info.SpecialKind == ARMInstrInfo::special_LoadLiteral
			&& DecodeLiteral(thumb, instrs[i], literalAddr))
		{
			u32 translatedAddr = LocaliseCodeAddress(cpu->Num, literalAddr);
			if (!translatedAddr)
			{
				printf("literal in non executable memory?\n");
			}
			u32 translatedAddrRounded = translatedAddr & ~0x1FF;

			u32 j = 0;
			for (; j < numAddressRanges; j++)
				if (addressRanges[j] == translatedAddrRounded)
					break;
			if (j == numAddressRanges)
				addressRanges[numAddressRanges++] = translatedAddrRounded;
			addressMasks[j] |= 1 << ((translatedAddr & 0x1FF) / 16);
			JIT_DEBUGPRINT("literal loading %08x %08x %08x %08x\n", literalAddr, translatedAddr, addressMasks[j], addressRanges[j]);
			cpu->DataRead32(literalAddr, &literalValues[numLiterals]);
			literalLoadAddrs[numLiterals++] = translatedAddr;
		}

		if (thumb && instrs[i].Info.Kind == ARMInstrInfo::tk_BL_LONG_2 && i > 0
			&& instrs[i - 1].Info.Kind == ARMInstrInfo::tk_BL_LONG_1)
		{
			instrs[i - 1].Info.Kind = ARMInstrInfo::tk_BL_LONG;
			instrs[i - 1].Instr = (instrs[i - 1].Instr & 0xFFFF) | (instrs[i].Instr << 16);
			instrs[i - 1].Info.DstRegs = 0xC000;
			instrs[i - 1].Info.SrcRegs = 0;
			instrs[i - 1].Info.EndBlock = true;
			i--;
		}

		if (instrs[i].Info.Branches() && Config::JIT_BranchOptimisations)
		{
			bool hasBranched = cpu->R[15] != r15;

			bool link;
			u32 cond, target, linkAddr;
			bool staticBranch = DecodeBranch(thumb, instrs[i], cond, hasLink, lr, link, linkAddr, target);
			JIT_DEBUGPRINT("branch cond %x target %x (%d)\n", cond, target, hasBranched);

			if (staticBranch)
			{
				instrs[i].BranchFlags |= branch_StaticTarget;

				bool isBackJump = false;
				if (hasBranched)
				{
					for (int j = 0; j < i; j++)
					{
						if (instrs[i].Addr == target)
						{
							isBackJump = true;
							break;
						}
					}
				}

				if (cond < 0xE && target < instrs[i].Addr && target >= lastSegmentStart)
				{
					// we might have an idle loop
					u32 backwardsOffset = (instrs[i].Addr - target) / (thumb ? 2 : 4);
					if (IsIdleLoop(&instrs[i - backwardsOffset], backwardsOffset + 1))
					{
						instrs[i].BranchFlags |= branch_IdleBranch;
						JIT_DEBUGPRINT("found %s idle loop %d in block %x\n", thumb ? "thumb" : "arm", cpu->Num, blockAddr);
					}
				}
				else if (hasBranched && !isBackJump && i + 1 < Config::JIT_MaxBlockSize)
				{
					if (link)
					{
						lr = linkAddr;
						hasLink = true;
					}
					
					r15 = target + (thumb ? 2 : 4);
					assert(r15 == cpu->R[15]);

					JIT_DEBUGPRINT("block lengthened by static branch (target %x)\n", target);

					nextInstr[0] = cpu->NextInstr[0];
					nextInstr[1] = cpu->NextInstr[1];

					nextInstrAddr[0] = target;
					nextInstrAddr[1] = r15;

					lastSegmentStart = target;

					instrs[i].Info.EndBlock = false;

					if (cond < 0xE)
						instrs[i].BranchFlags |= branch_FollowCondTaken;
				}
			}

			if (!hasBranched && cond < 0xE && i + 1 < Config::JIT_MaxBlockSize)
			{
				instrs[i].Info.EndBlock = false;
				instrs[i].BranchFlags |= branch_FollowCondNotTaken;
			}
		}

        i++;

		bool canCompile = JITCompiler->CanCompile(thumb, instrs[i - 1].Info.Kind);
		bool secondaryFlagReadCond = !canCompile || (instrs[i - 1].BranchFlags & (branch_FollowCondTaken | branch_FollowCondNotTaken));
		if (instrs[i - 1].Info.ReadFlags != 0 || secondaryFlagReadCond)
			FloodFillSetFlags(instrs, i - 2, !secondaryFlagReadCond ? instrs[i - 1].Info.ReadFlags : 0xF);
    } while(!instrs[i - 1].Info.EndBlock && i < Config::JIT_MaxBlockSize && !cpu->Halted && (!cpu->IRQ || (cpu->CPSR & 0x80)));

	u32 literalHash = (u32)XXH3_64bits(literalValues, numLiterals * 4);
	u32 instrHash = (u32)XXH3_64bits(instrValues, i * 4);

	JitBlock* prevBlock = RestoreCandidates.LookUp(instrHash);
	bool mayRestore = true;
	if (prevBlock)
	{
		RestoreCandidates.Remove(instrHash);

		mayRestore = prevBlock->StartAddr == blockAddr && prevBlock->LiteralHash == literalHash;

		if (mayRestore && prevBlock->NumAddresses == numAddressRanges)
		{
			for (int j = 0; j < numAddressRanges; j++)
			{
				if (prevBlock->AddressRanges()[j] != addressRanges[j]
					|| prevBlock->AddressMasks()[j] != addressMasks[j])
				{
					mayRestore = false;
					break;
				}
			}
		}
		else
			mayRestore = false;
	}
	else
	{
		mayRestore = false;
		prevBlock = NULL;
	}

	JitBlock* block;
	if (!mayRestore)
	{
		if (prevBlock)
			delete prevBlock;

		block = new JitBlock(cpu->Num, i, numAddressRanges, numLiterals);
		block->LiteralHash = literalHash;
		block->InstrHash = instrHash;
		for (int j = 0; j < numAddressRanges; j++)
			block->AddressRanges()[j] = addressRanges[j];
		for (int j = 0; j < numAddressRanges; j++)
			block->AddressMasks()[j] = addressMasks[j];
		for (int j = 0; j < numLiterals; j++)
			block->Literals()[j] = literalLoadAddrs[j];

		block->StartAddr = blockAddr;
		block->StartAddrLocal = localAddr;

		FloodFillSetFlags(instrs, i - 1, 0xF);

		block->EntryPoint = JITCompiler->CompileBlock(cpu, thumb, instrs, i);

		JIT_DEBUGPRINT("block start %p\n", block->EntryPoint);
	}
	else
	{
		JIT_DEBUGPRINT("restored! %p\n", prevBlock);
		block = prevBlock;
	}

	for (int j = 0; j < numAddressRanges; j++)
	{
		assert(addressRanges[j] == block->AddressRanges()[j]);
		assert(addressMasks[j] == block->AddressMasks()[j]);
		assert(addressMasks[j] != 0);

		AddressRange* region = CodeMemRegions[addressRanges[j] >> 27];

		if (!PageContainsCode(&region[(addressRanges[j] & 0x7FFF000) / 512]))
			ARMJIT_Memory::SetCodeProtection(addressRanges[j] >> 27, addressRanges[j] & 0x7FFFFFF, true);

		AddressRange* range = &region[(addressRanges[j] & 0x7FFFFFF) / 512];
		range->Code |= addressMasks[j];
		range->Blocks.Add(block);
	}

	if (cpu->Num == 0)
		JitBlocks9[blockAddr] = block;
	else
		JitBlocks7[blockAddr] = block;

	u64* entry = &FastBlockLookupRegions[(localAddr >> 27)][(localAddr & 0x7FFFFFF) / 2];
	*entry = ((u64)blockAddr | cpu->Num) << 32;
	*entry |= JITCompiler->SubEntryOffset(block->EntryPoint);
}

void InvalidateByAddr(u32 localAddr)
{
	JIT_DEBUGPRINT("invalidating by addr %x\n", localAddr);

	AddressRange* region = CodeMemRegions[localAddr >> 27];
	AddressRange* range = &region[(localAddr & 0x7FFFFFF) / 512];
	u32 mask = 1 << ((localAddr & 0x1FF) / 16);

	range->Code = 0;
	for (int i = 0; i < range->Blocks.Length;)
	{
		JitBlock* block = range->Blocks[i];

		bool invalidated = false;
		u32 mask = 0;
		for (int j = 0; j < block->NumAddresses; j++)
		{
			if (block->AddressRanges()[j] == (localAddr & ~0x1FF))
			{
				mask = block->AddressMasks()[j];
				invalidated = block->AddressMasks()[j] & mask;
				break;
			}
		}
		assert(mask);
		if (!invalidated)
		{
			range->Code |= mask;
			i++;
			continue;
		}
		range->Blocks.Remove(i);

		if (range->Blocks.Length == 0
			&& !PageContainsCode(&region[(localAddr & 0x7FFF000) / 512]))
		{
			ARMJIT_Memory::SetCodeProtection(localAddr >> 27, localAddr & 0x7FFFFFF, false);
		}

		bool literalInvalidation = false;
		for (int j = 0; j < block->NumLiterals; j++)
		{
			u32 addr = block->Literals()[j];
			if (addr == localAddr)
			{
				if (InvalidLiterals.Find(localAddr) != -1)
				{
					InvalidLiterals.Add(localAddr);
					JIT_DEBUGPRINT("found invalid literal %d\n", InvalidLiterals.Length);
				}
				literalInvalidation = true;
				break;
			}
		}
		for (int j = 0; j < block->NumAddresses; j++)
		{
			u32 addr = block->AddressRanges()[j];
			if ((addr / 512) != (localAddr / 512))
			{
				AddressRange* otherRegion = CodeMemRegions[addr >> 27];
				AddressRange* otherRange = &otherRegion[(addr & 0x7FFFFFF) / 512];
				assert(otherRange != range);

				bool removed = otherRange->Blocks.RemoveByValue(block);
				assert(removed);

				if (otherRange->Blocks.Length == 0)
				{
					if (!PageContainsCode(&otherRegion[(addr & 0x7FFF000) / 512]))
						ARMJIT_Memory::SetCodeProtection(addr >> 27, addr & 0x7FFFFFF, false);

					otherRange->Code = 0;
				}
			}
		}

		FastBlockLookupRegions[block->StartAddrLocal >> 27][(block->StartAddrLocal & 0x7FFFFFF) / 2] = (u64)UINT32_MAX << 32;
		if (block->Num == 0)
			JitBlocks9.erase(block->StartAddr);
		else
			JitBlocks7.erase(block->StartAddr);

		if (!literalInvalidation)
		{
			JitBlock* prevBlock = RestoreCandidates.Insert(block->InstrHash, block);
			if (prevBlock)
				delete prevBlock;
		}
		else
		{
			delete block;
		}
	}
}

void CheckAndInvalidateITCM()
{
	for (u32 i = 0; i < ITCMPhysicalSize; i+=16)
	{
		if (CodeIndexITCM[i / 512].Code & (1 << ((i & 0x1FF) / 16)))
		{
			InvalidateByAddr(i | (ARMJIT_Memory::memregion_ITCM << 27));
		}
	}
}

template <u32 num, int region>
void CheckAndInvalidate(u32 addr)
{
	u32 localAddr = ARMJIT_Memory::LocaliseAddress(region, num, addr);
	if (CodeMemRegions[region][(localAddr & 0x7FFFFFF) / 512].Code & (1 << ((localAddr & 0x1FF) / 16)))
		InvalidateByAddr(localAddr);
}

JitBlockEntry LookUpBlock(u32 num, u64* entries, u32 offset, u32 addr)
{
	u64* entry = &entries[offset / 2];
	if (*entry >> 32 == (addr | num))
		return JITCompiler->AddEntryOffset((u32)*entry);
	return NULL;
}

void blockSanityCheck(u32 num, u32 blockAddr, JitBlockEntry entry)
{
	u32 localAddr = LocaliseCodeAddress(num, blockAddr);
	assert(JITCompiler->AddEntryOffset((u32)FastBlockLookupRegions[localAddr >> 27][(localAddr & 0x7FFFFFF) / 2]) == entry);
}

bool SetupExecutableRegion(u32 num, u32 blockAddr, u64*& entry, u32& start, u32& size)
{
	// amazingly ignoring the DTCM is the proper behaviour for code fetches
	int region = num == 0
		? ARMJIT_Memory::ClassifyAddress9(blockAddr)
		: ARMJIT_Memory::ClassifyAddress7(blockAddr);

	u32 memoryOffset;
	if (FastBlockLookupRegions[region]
		&& ARMJIT_Memory::GetMirrorLocation(region, num, blockAddr, memoryOffset, start, size))
	{
		//printf("setup exec region %d %d %08x %08x %x %x\n", num, region, blockAddr, start, size, memoryOffset);
		entry = FastBlockLookupRegions[region] + memoryOffset / 2;
		return true;
	}
	return false;
}

template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_MainRAM>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_MainRAM>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_SharedWRAM>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_SharedWRAM>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_WRAM7>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_VWRAM>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_VRAM>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_ITCM>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_NewSharedWRAM_A>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_NewSharedWRAM_A>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_NewSharedWRAM_B>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_NewSharedWRAM_B>(u32);
template void CheckAndInvalidate<0, ARMJIT_Memory::memregion_NewSharedWRAM_C>(u32);
template void CheckAndInvalidate<1, ARMJIT_Memory::memregion_NewSharedWRAM_C>(u32);

void ResetBlockCache()
{
	printf("Resetting JIT block cache...\n");

	InvalidLiterals.Clear();
	for (int i = 0; i < ARMJIT_Memory::memregions_Count; i++)
		memset(FastBlockLookupRegions[i], 0xFF, CodeRegionSizes[i] * sizeof(u64) / 2);
	RestoreCandidates.Reset();
	for (int i = 0; i < sizeof(RestoreCandidates.Table)/sizeof(RestoreCandidates.Table[0]); i++)
	{
		if (RestoreCandidates.Table[i].ValA)
		{
			delete RestoreCandidates.Table[i].ValA;
			RestoreCandidates.Table[i].ValA = NULL;
		}
		if (RestoreCandidates.Table[i].ValA)
		{
			delete RestoreCandidates.Table[i].ValB;
			RestoreCandidates.Table[i].ValB = NULL;
		}
	}
	for (auto it : JitBlocks9)
	{
		JitBlock* block = it.second;
		for (int j = 0; j < block->NumAddresses; j++)
		{
			u32 addr = block->AddressRanges()[j];
			AddressRange* range = &CodeMemRegions[addr >> 27][(addr & 0x7FFFFFF) / 512];
			range->Blocks.Clear();
			range->Code = 0;
		}
		delete block;
	}
	for (auto it : JitBlocks7)
	{
		JitBlock* block = it.second;
		for (int j = 0; j < block->NumAddresses; j++)
		{
			u32 addr = block->AddressRanges()[j];
			AddressRange* range = &CodeMemRegions[addr >> 27][(addr & 0x7FFFFFF) / 512];
			range->Blocks.Clear();
			range->Code = 0;
		}
	}
	JitBlocks9.clear();
	JitBlocks7.clear();

	JITCompiler->Reset();
}

}
