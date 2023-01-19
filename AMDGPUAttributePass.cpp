//=============================================================================
// FILE:
//    AMDGPUAttributePass.cpp
//
// DESCRIPTION:
//    Find device functions targeting the AMD GPU architecture and set different
//    values for AMD GPU specific attributes.
//
//=============================================================================

#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace omp;

//-----------------------------------------------------------------------------
// AMDGPUAttributePass implementation
//-----------------------------------------------------------------------------
// No need to expose the internals of the pass to the outside world - keep
// everything in an anonymous namespace.
namespace {

// This method implements what the pass does
void visitor(Module &M) {
  OpenMPIRBuilder OMPBuilder(M);
  OMPBuilder.initialize();

  SmallPtrSet<Function *, 8> KernelEntryFunctions;

  auto IsKernelEntry = [&](Function &F) {
    FunctionCallee KernelInit = OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_target_init);
    FunctionCallee KernelDeinit = OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_target_deinit);

    bool CallsTargetInit = false;
    bool CallsTargetDeinit = false;

    for(Use &U : KernelInit.getCallee()->uses())
      if(auto *I = dyn_cast<Instruction>(U.getUser()))
        if(I->getFunction() == &F) {
          CallsTargetInit = true;
          break;
        }

    for(Use &U : KernelDeinit.getCallee()->uses())
      if(Instruction *I = dyn_cast<Instruction>(U.getUser()))
        if(I->getFunction() == &F) {
          CallsTargetDeinit = true;
          break;
        }

    return (CallsTargetInit && CallsTargetDeinit);
  };
  
  for(Function &F : M) 
    if(IsKernelEntry(F))
      KernelEntryFunctions.insert(&F);

  for(Function *F : KernelEntryFunctions) {
    outs() << "Kernel entry function " << F->getName() << "\n";
    F->addFnAttr();
  }
    
}

// New PM implementation
struct AMDGPUAttributePass : PassInfoMixin<AMDGPUAttributePass> {
  // Main entry point, takes IR unit to run the pass on (&F) and the
  // corresponding pass manager (to be queried if need be)
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    visitor(M);
    // TODO: is anything preserved?
    return PreservedAnalyses::none();
    //return PreservedAnalyses::all();
  }

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// Legacy PM implementation
struct LegacyAMDGPUAttributePass : public ModulePass {
  static char ID;
  LegacyAMDGPUAttributePass() : ModulePass(ID) {}
  // Main entry point - the name conveys what unit of IR this is to be run on.
  bool runOnModule(Module &M) override {
    visitor(M);

    // TODO: what is preserved?
    return true;
    // Doesn't modify the input unit of IR, hence 'false'
    //return false;
  }
};
} // namespace

//-----------------------------------------------------------------------------
// New PM Registration
//-----------------------------------------------------------------------------
llvm::PassPluginLibraryInfo getAMDGPUAttributePassPluginInfo() {
  const auto callback = [](PassBuilder &PB) {
    // TODO: decide where to insert it in the pipeline. Early avoids
    // inlining jit function (which disables jit'ing) but may require more
    // optimization, hence overhead, at runtime.
    PB.registerPipelineStartEPCallback([&](ModulePassManager &MPM, auto) {
    //PB.registerPipelineEarlySimplificationEPCallback([&](ModulePassManager &MPM, auto) {
    // XXX: LastEP can break jit'ing, jit function is inlined!
    //PB.registerOptimizerLastEPCallback([&](ModulePassManager &MPM, auto) {
      MPM.addPass(AMDGPUAttributePass());
      return true;
    });
  };

  return {LLVM_PLUGIN_API_VERSION, "AMDGPUAttributePass", LLVM_VERSION_STRING, callback};
}

// TODO: use by jit-pass name.
// This is the core interface for pass plugins. It guarantees that 'opt' will
// be able to recognize AMDGPUAttributePass when added to the pass pipeline on the
// command line, i.e. via '-passes=jit-pass'
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getAMDGPUAttributePassPluginInfo();
}

//-----------------------------------------------------------------------------
// Legacy PM Registration
//-----------------------------------------------------------------------------
// The address of this variable is used to uniquely identify the pass. The
// actual value doesn't matter.
char LegacyAMDGPUAttributePass::ID = 0;

// This is the core interface for pass plugins. It guarantees that 'opt' will
// recognize LegacyAMDGPUAttributePass when added to the pass pipeline on the command
// line, i.e.  via '--legacy-jit-pass'
static RegisterPass<LegacyAMDGPUAttributePass>
    X("legacy-amdgpuattribute-pass", "AMDGPU Attribute Pass",
      false, // This pass doesn't modify the CFG => false
      false // This pass is not a pure analysis pass => false
    );
