# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


Parallel Check
```
MAP
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (163) 
-----------------------------------------------------------------------------|loop #ID
    def _map(                                                                | 
        out: Storage,                                                        | 
        out_shape: Shape,                                                    | 
        out_strides: Strides,                                                | 
        in_storage: Storage,                                                 | 
        in_shape: Shape,                                                     | 
        in_strides: Strides,                                                 | 
    ) -> None:                                                               | 
      # Check if tensors are stride-aligned for fast path                    | 
        is_aligned = True                                                    | 
        for i in range(len(out_shape)):                                      | 
            if (out_strides[i] != in_strides[i] or                           | 
                out_shape[i] != in_shape[i]):                                | 
                is_aligned = False                                           | 
                break                                                        | 
                                                                             | 
        if is_aligned:                                                       | 
            # Main parallel loop if stride-aligned                           | 
            for i in prange(out.size):---------------------------------------| #0
                out[i] = fn(in_storage[i])                                   | 
        else:                                                                | 
            # Main parallel loop                                             | 
            for i in prange(out.size):---------------------------------------| #1
                # All indices use numpy buffers                              | 
                out_index = np.empty(MAX_DIMS, np.int32)                     | 
                in_index = np.empty(MAX_DIMS, np.int32)                      | 
                to_index(i, out_shape, out_index)                            | 
                broadcast_index(out_index, out_shape, in_shape, in_index)    | 
                in_position = index_to_position(in_index, in_strides)        | 
                out_position = index_to_position(out_index, out_strides)     | 
                out[out_position] = fn(in_storage[in_position])              | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (187) is hoisted out
 of the parallel loop labelled #1 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (188) is hoisted out
 of the parallel loop labelled #1 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (222)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (222) 
---------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                    | 
        out: Storage,                                                            | 
        out_shape: Shape,                                                        | 
        out_strides: Strides,                                                    | 
        a_storage: Storage,                                                      | 
        a_shape: Shape,                                                          | 
        a_strides: Strides,                                                      | 
        b_storage: Storage,                                                      | 
        b_shape: Shape,                                                          | 
        b_strides: Strides,                                                      | 
    ) -> None:                                                                   | 
                                                                                 | 
                                                                                 | 
        # Check if tensors are stride-aligned (moved outside parallel region)    | 
        is_aligned = True                                                        | 
        for i in range(len(out_shape)):                                          | 
            if (out_strides[i] != a_strides[i] or                                | 
                out_strides[i] != b_strides[i] or                                | 
                out_shape[i] != a_shape[i] or                                    | 
                out_shape[i] != b_shape[i]):                                     | 
                is_aligned = False                                               | 
                break                                                            | 
                                                                                 | 
        if is_aligned:                                                           | 
            # Main parallel loop if stride-aligned                               | 
            for i in prange(len(out)):-------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                          | 
        else:                                                                    | 
        # Main parallel loop                                                     | 
            for i in prange(len(out)):     --------------------------------------| #3
                # All indices use numpy buffers                                  | 
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                   | 
                a_index = np.empty(MAX_DIMS, dtype=np.int32)                     | 
                b_index = np.empty(MAX_DIMS, dtype=np.int32)                     | 
                to_index(i, out_shape, out_index)                                | 
                broadcast_index(out_index, out_shape, a_shape, a_index)          | 
                broadcast_index(out_index, out_shape, b_shape, b_index)          | 
                j = index_to_position(a_index, a_strides)                        | 
                k = index_to_position(b_index, b_strides)                        | 
                out[i] = fn(a_storage[j], b_storage[k])                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (253) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (254) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (255) is hoisted out
 of the parallel loop labelled #3 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (287)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (287) 
-----------------------------------------------------------------------|loop #ID
    def _reduce(                                                       | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        reduce_dim: int,                                               | 
    ) -> None:                                                         | 
                                                                       | 
                                                                       | 
        size = len(out)                                                | 
        reduce_stride = a_strides[reduce_dim]                          | 
                                                                       | 
        for out_pos in prange(size):-----------------------------------| #4
            # Outer loop: calculate indices                            | 
            out_index = np.empty(MAX_DIMS, np.int32)                   | 
            to_index(out_pos, out_shape, out_index)                    | 
            base_position = index_to_position(out_index, a_strides)    | 
                                                                       | 
            # Inner loop: only local variables and operations          | 
            val = out[out_pos]                                         | 
            pos = base_position                                        | 
            for i in range(a_shape[reduce_dim]):                       | 
                val = fn(val, a_storage[pos])                          | 
                pos += reduce_stride                                   | 
                                                                       | 
            out[out_pos] = val                                         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (303) is hoisted out
 of the parallel loop labelled #4 (it will be performed before the loop is 
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (319)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/zzy/Desktop/CS5781/mod3-zhubzy/minitorch/fast_ops.py (319) 
--------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                                                        | 
    out: Storage,                                                                                                                                                   | 
    out_shape: Shape,                                                                                                                                               | 
    out_strides: Strides,                                                                                                                                           | 
    a_storage: Storage,                                                                                                                                             | 
    a_shape: Shape,                                                                                                                                                 | 
    a_strides: Strides,                                                                                                                                             | 
    b_storage: Storage,                                                                                                                                             | 
    b_shape: Shape,                                                                                                                                                 | 
    b_strides: Strides,                                                                                                                                             | 
) -> None:                                                                                                                                                          | 
    """NUMBA tensor matrix multiply function.                                                                                                                       | 
                                                                                                                                                                    | 
    Should work for any tensor shapes that broadcast as long as                                                                                                     | 
                                                                                                                                                                    | 
    ```                                                                                                                                                             | 
    assert a_shape[-1] == b_shape[-2]                                                                                                                               | 
    ```                                                                                                                                                             | 
                                                                                                                                                                    | 
    Optimizations:                                                                                                                                                  | 
                                                                                                                                                                    | 
    * Outer loop in parallel                                                                                                                                        | 
    * No index buffers or function calls                                                                                                                            | 
    * Inner loop should have no global writes, 1 multiply.                                                                                                          | 
                                                                                                                                                                    | 
                                                                                                                                                                    | 
    Args:                                                                                                                                                           | 
    ----                                                                                                                                                            | 
        out (Storage): storage for `out` tensor                                                                                                                     | 
        out_shape (Shape): shape for `out` tensor                                                                                                                   | 
        out_strides (Strides): strides for `out` tensor                                                                                                             | 
        a_storage (Storage): storage for `a` tensor                                                                                                                 | 
        a_shape (Shape): shape for `a` tensor                                                                                                                       | 
        a_strides (Strides): strides for `a` tensor                                                                                                                 | 
        b_storage (Storage): storage for `b` tensor                                                                                                                 | 
        b_shape (Shape): shape for `b` tensor                                                                                                                       | 
        b_strides (Strides): strides for `b` tensor                                                                                                                 | 
                                                                                                                                                                    | 
    Returns:                                                                                                                                                        | 
    -------                                                                                                                                                         | 
        None : Fills in `out`                                                                                                                                       | 
                                                                                                                                                                    | 
    """                                                                                                                                                             | 
                                                                                                                                                                    | 
                                                                                                                                                                    | 
                                                                                                                                                                    | 
                                                                                                                                                                    | 
    a_rows = a_shape[-2]                                                                                                                                            | 
    a_cols = a_shape[-1]                                                                                                                                            | 
    b_cols = b_shape[-1]                                                                                                                                            | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                                                          | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                                                          | 
    out_batch_stride = out_strides[0] if out_shape[0] > 1 else 0                                                                                                    | 
                                                                                                                                                                    | 
                                                                                                                                                                    | 
    # Handle batch dimensions with parallel outer loop                                                                                                              | 
    for batch in prange(max(1, out_shape[0])):----------------------------------------------------------------------------------------------------------------------| #8
        # Calculate base offsets for this batch                                                                                                                     | 
        a_batch_offset = batch * a_batch_stride                                                                                                                     | 
        b_batch_offset = batch * b_batch_stride                                                                                                                     | 
        out_batch_offset = batch * out_batch_stride                                                                                                                 | 
                                                                                                                                                                    | 
        # Core matrix multiplication for this batch                                                                                                                 | 
        for i in prange(a_rows):------------------------------------------------------------------------------------------------------------------------------------| #7
            for j in prange(b_cols):--------------------------------------------------------------------------------------------------------------------------------| #6
                # Initialize sum for dot product                                                                                                                    | 
                acc = 0.0                                                                                                                                           | 
                # Single inner loop for dot product between row of A and column of B                                                                                | 
                for k in prange(a_cols):----------------------------------------------------------------------------------------------------------------------------| #5
                    acc += a_storage[a_batch_offset + i * a_strides[-2] + k * a_strides[-1]] * b_storage[b_batch_offset + k * b_strides[-2] + j * b_strides[-1]]    | 
                                                                                                                                                                    | 
                out[out_batch_offset + i * out_strides[-2] + j * out_strides[-1]] = acc                                                                             | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #8, #7).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--7 --> rewritten as a serial loop
      +--6 --> rewritten as a serial loop
         +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (parallel)
      +--6 (parallel)
         +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--7 (serial)
      +--6 (serial)
         +--5 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
```




```
(base) (.venv) zzy@Zachs-MacBook-Pro mod3-zhubzy % python  project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.04
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
Epoch  0  loss  56.54978760980882 correct 35
Time  7.084343910217285
Epoch  10  loss  27.662637879473046 correct 42
Time  0.06436395645141602
Epoch  20  loss  22.695642902663252 correct 44
Time  0.06461024284362793
Epoch  30  loss  21.24093587576185 correct 42
Time  0.0682370662689209
Epoch  40  loss  18.722721651669843 correct 44
Time  0.06739997863769531
Epoch  50  loss  17.29844577854628 correct 45
Time  0.06832408905029297
Epoch  60  loss  16.962977910080184 correct 46
Time  0.0704050064086914
Epoch  70  loss  15.32692966833259 correct 45
Time  0.06202411651611328
Epoch  80  loss  13.146006353986076 correct 45
Time  0.06833791732788086
Epoch  90  loss  14.5207310778544 correct 45
Time  0.07681608200073242
Epoch  100  loss  11.538811245258414 correct 49
Time  0.1645498275756836
Epoch  110  loss  11.88336409518752 correct 47
Time  0.07569384574890137
Epoch  120  loss  9.961885977174855 correct 47
Time  0.07174992561340332
Epoch  130  loss  9.842618257880106 correct 47
Time  0.07079815864562988
Epoch  140  loss  9.008338500938832 correct 50
Time  0.07600688934326172
Epoch  150  loss  8.387115792050858 correct 50
Time  0.08619427680969238
Epoch  160  loss  8.22141892613131 correct 46
Time  0.06826114654541016
Epoch  170  loss  6.861214381559794 correct 49
Time  0.07049393653869629
Epoch  180  loss  8.850211526043442 correct 48
Time  0.08052802085876465
Epoch  190  loss  7.376256505811515 correct 49
Time  0.07501721382141113
Epoch  200  loss  7.852551721205447 correct 49
Time  0.06364202499389648
Epoch  210  loss  6.040633956620672 correct 50
Time  0.07879805564880371
Epoch  220  loss  5.63268554119321 correct 47
Time  0.08746767044067383
Epoch  230  loss  5.174508758272623 correct 49
Time  0.09369707107543945
Epoch  240  loss  4.857614111319829 correct 49
Time  0.07102417945861816
Epoch  250  loss  4.8359399846736535 correct 49
Time  0.07375216484069824
Epoch  260  loss  5.209504609142986 correct 50
Time  0.19364404678344727
Epoch  270  loss  8.035562065395776 correct 48
Time  0.11053013801574707
Epoch  280  loss  4.27419895774245 correct 49
Time  0.11304616928100586
Epoch  290  loss  5.151967021147488 correct 49
Time  0.10803103446960449
Epoch  300  loss  5.744063568753005 correct 47
Time  0.11842012405395508
Epoch  310  loss  4.921296761940698 correct 50
Time  0.12329792976379395
Epoch  320  loss  5.331160820858481 correct 50
Time  0.06521391868591309
Epoch  330  loss  3.844260733992545 correct 50
Time  0.11047482490539551
Epoch  340  loss  4.573925654468451 correct 50
Time  0.11349725723266602
Epoch  350  loss  3.9048874277630716 correct 50
Time  0.10952186584472656
Epoch  360  loss  3.283699227313102 correct 49
Time  0.07077717781066895
Epoch  370  loss  3.495472195693642 correct 50
Time  0.06808805465698242
Epoch  380  loss  3.0078434313210205 correct 50
Time  0.06714916229248047
Epoch  390  loss  3.040959597314893 correct 49
Time  0.0641469955444336
Epoch  400  loss  4.076027357421617 correct 50
Time  0.07624411582946777
Epoch  410  loss  3.832657159207654 correct 50
Time  0.06606698036193848
Epoch  420  loss  3.1014826242537215 correct 50
Time  0.06345605850219727
Epoch  430  loss  3.2461970566952227 correct 50
Time  0.06995010375976562
Epoch  440  loss  3.52456283402348 correct 50
Time  0.07509183883666992
Epoch  450  loss  2.849015698712125 correct 49
Time  0.06976008415222168
Epoch  460  loss  3.733841403404448 correct 49
Time  0.06329822540283203
Epoch  470  loss  2.931923135428612 correct 50
Time  0.06467723846435547
Epoch  480  loss  2.5820387022543043 correct 50
Time  0.07248210906982422
Epoch  490  loss  2.7244590590654 correct 50
Time  0.06610417366027832
```



### Split (Cuda)
```
Epoch  0  loss  31.384926076832468 correct 33
Time  4.134276628494263
Epoch  10  loss  27.243605283561905 correct 35
Time  1.3737156391143799
Epoch  20  loss  24.928284159502944 correct 33
Time  1.3707046508789062
Epoch  30  loss  23.735637977440575 correct 43
Time  1.713878870010376
Epoch  40  loss  17.645568942010716 correct 44
Time  1.3844807147979736
Epoch  50  loss  14.817267008574657 correct 46
Time  1.3785765171051025
Epoch  60  loss  14.215970661759584 correct 46
Time  1.402282953262329
Epoch  70  loss  12.454394481880062 correct 48
Time  1.3551123142242432
Epoch  80  loss  10.969147007363782 correct 47
Time  1.8401472568511963
Epoch  90  loss  9.80696747880413 correct 47
Time  1.4165105819702148
Epoch  100  loss  9.321443410800192 correct 48
Time  1.436136245727539
Epoch  110  loss  9.015034722453978 correct 47
Time  1.3732733726501465
Epoch  120  loss  7.43809101203226 correct 48
Time  1.370556354522705
Epoch  130  loss  7.454355404410424 correct 47
Time  2.0077431201934814
Epoch  140  loss  7.073189916615148 correct 48
Time  1.3712921142578125
Epoch  150  loss  7.527047920483954 correct 48
Time  1.3657023906707764
Epoch  160  loss  7.240165301286256 correct 48
Time  1.3794848918914795
Epoch  170  loss  6.048516825091017 correct 48
Time  1.359574556350708
Epoch  180  loss  6.572403711367681 correct 49
Time  2.075929880142212
Epoch  190  loss  5.617572887803409 correct 48
Time  1.3708415031433105
Epoch  200  loss  6.465653689844039 correct 48
Time  1.364954948425293
Epoch  210  loss  5.319666576606373 correct 49
Time  1.3513641357421875
Epoch  220  loss  5.692061592998051 correct 49
Time  1.4182348251342773
Epoch  230  loss  4.618108700316135 correct 49
Time  2.1561052799224854
Epoch  240  loss  4.555122233794648 correct 49
Time  1.4459478855133057
Epoch  250  loss  4.8034141908490415 correct 49
Time  1.3719429969787598
Epoch  260  loss  4.555810717227082 correct 48
Time  1.3791284561157227
Epoch  270  loss  4.801902563602609 correct 49
Time  1.3712406158447266
Epoch  280  loss  4.185735351396436 correct 49
Time  1.9357869625091553
Epoch  290  loss  4.762782411910719 correct 49
Time  1.3552429676055908
Epoch  300  loss  3.857647896278166 correct 48
Time  1.3685460090637207
Epoch  310  loss  3.8350143586407808 correct 49
Time  1.3741211891174316
Epoch  320  loss  3.623835953440258 correct 49
Time  1.3748514652252197
Epoch  330  loss  3.696639059207575 correct 49
Time  1.8626155853271484
Epoch  340  loss  3.3287932397639617 correct 49
Time  1.3792424201965332
Epoch  350  loss  4.578047113249321 correct 49
Time  1.4445393085479736
Epoch  360  loss  4.654515839416049 correct 49
Time  1.4536397457122803
Epoch  370  loss  4.275922914662182 correct 48
Time  1.5478136539459229
Epoch  380  loss  3.6806461679484244 correct 50
Time  1.6907322406768799
Epoch  390  loss  3.3266007045010957 correct 49
Time  1.3561546802520752
Epoch  400  loss  3.8179453633989047 correct 50
Time  1.4045915603637695
Epoch  410  loss  3.6114261503356055 correct 49
Time  1.3562486171722412
Epoch  420  loss  3.2840762543354574 correct 49
Time  1.6746196746826172
Epoch  430  loss  3.490976537580467 correct 49
Time  1.3744444847106934
Epoch  440  loss  3.31310822847161 correct 50
Time  1.3573658466339111
Epoch  450  loss  3.2260136602034652 correct 49
Time  1.3727030754089355
Epoch  460  loss  3.7747496442196624 correct 49
Time  1.364459753036499
Epoch  470  loss  2.7794892600806493 correct 49
Time  1.7031469345092773
Epoch  480  loss  2.4208461048678385 correct 49
Time  1.4443893432617188
Epoch  490  loss  5.152384305272025 correct 50
Time  1.4254779815673828
```

### Xor (Cuda)
```
Epoch  0  loss  34.10347874918497 correct 28
Time  4.272539377212524
Epoch  10  loss  27.9609849446496 correct 41
Time  1.4196641445159912
Epoch  20  loss  25.17458043324409 correct 41
Time  1.3557345867156982
Epoch  30  loss  24.061583029972166 correct 40
Time  1.8856303691864014
Epoch  40  loss  21.169316905150907 correct 42
Time  1.3473100662231445
Epoch  50  loss  19.93585033267518 correct 43
Time  1.356959581375122
Epoch  60  loss  18.23804780104609 correct 43
Time  1.3398339748382568
Epoch  70  loss  17.388144493285928 correct 42
Time  1.391139030456543
Epoch  80  loss  17.323131331152503 correct 42
Time  1.9043605327606201
Epoch  90  loss  16.174343838529644 correct 43
Time  1.409665822982788
Epoch  100  loss  15.890874644713357 correct 43
Time  1.4098868370056152
Epoch  110  loss  15.26083351410884 correct 43
Time  1.3502919673919678
Epoch  120  loss  14.182149808800702 correct 43
Time  1.3445734977722168
Epoch  130  loss  13.67806595736242 correct 43
Time  1.5884571075439453
Epoch  140  loss  13.91529202619027 correct 43
Time  1.487384557723999
Epoch  150  loss  12.485949307804468 correct 43
Time  1.3480346202850342
Epoch  160  loss  12.958781011709092 correct 44
Time  1.3477070331573486
Epoch  170  loss  11.662569857486943 correct 45
Time  1.3415470123291016
Epoch  180  loss  10.728635076792761 correct 44
Time  1.3395240306854248
Epoch  190  loss  10.702409899999799 correct 44
Time  1.8379559516906738
Epoch  200  loss  10.20955018023777 correct 45
Time  1.3493919372558594
Epoch  210  loss  9.727683915924281 correct 47
Time  1.3532960414886475
Epoch  220  loss  9.05673804799224 correct 46
Time  1.4290673732757568
Epoch  230  loss  9.263143538650425 correct 45
Time  1.396369457244873
Epoch  240  loss  7.55973346854307 correct 46
Time  2.1454358100891113
Epoch  250  loss  7.942825165275456 correct 48
Time  1.3421387672424316
Epoch  260  loss  6.4911106193143 correct 47
Time  1.3713269233703613
Epoch  270  loss  7.383499613418968 correct 48
Time  1.3388690948486328
Epoch  280  loss  6.150704546152516 correct 50
Time  1.341841220855713
Epoch  290  loss  5.375980188220001 correct 49
Time  1.7036423683166504
Epoch  300  loss  5.177699583140969 correct 50
Time  1.3505511283874512
Epoch  310  loss  5.5931409172990225 correct 49
Time  1.327744960784912
Epoch  320  loss  4.821400615463157 correct 49
Time  1.3515257835388184
Epoch  330  loss  4.314021734651691 correct 50
Time  1.3368282318115234
Epoch  340  loss  4.911787026914105 correct 50
Time  1.4049310684204102
Epoch  350  loss  3.8915989151269654 correct 50
Time  1.8228845596313477
Epoch  360  loss  3.4655426806632543 correct 50
Time  1.4225468635559082
Epoch  370  loss  3.721746645745412 correct 49
Time  1.3437223434448242
Epoch  380  loss  3.5358818252304145 correct 49
Time  1.355257272720337
Epoch  390  loss  4.352693450822615 correct 49
Time  1.3409101963043213
Epoch  400  loss  3.1319000507476087 correct 50
Time  2.0178537368774414
Epoch  410  loss  3.4829490515631467 correct 49
Time  1.3503599166870117
Epoch  420  loss  3.9845357264473633 correct 49
Time  1.3433036804199219
Epoch  430  loss  2.8580903324720794 correct 50
Time  1.3512001037597656
Epoch  440  loss  3.099914250845528 correct 50
Time  1.3644983768463135
Epoch  450  loss  3.292307727219368 correct 50
Time  1.5671021938323975
Epoch  460  loss  2.60816562462026 correct 50
Time  1.4288175106048584
Epoch  470  loss  2.2971458904777573 correct 50
Time  1.3943936824798584
Epoch  480  loss  2.4030599235595935 correct 50
Time  1.3981335163116455
Epoch  490  loss  2.11992034028144 correct 50
Time  1.3351004123687744
```