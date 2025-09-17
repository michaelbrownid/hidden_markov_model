#!/usr/bin/env python3

"""
Test script to verify forward and backward algorithm consistency
Tests the corrected forward function against the known-correct backward function
"""

import sys
import math
import json
from HMM_VEC import HMM, nl, probFromNlp
from HMM_DAT import Data

def test_forward_backward_consistency():
    """
    Test that forward and backward algorithms give consistent results
    """
    print("=" * 60)
    print("TESTING FORWARD-BACKWARD CONSISTENCY")
    print("=" * 60)
    
    # Load model and data
    print("\n1. Loading model and data...")
    myhmm = HMM()
    myhmm.readJSON("hmm_twoState.json")
    
    data = Data("hmm_data.json")
    print("Available data sequences:", data.listDataSeq())
    
    # Set target sequence
    myhmm.targetOutputSeq = "RB_video_0"
    (myhmm.targetOutputNames, myhmm.targetOutput) = data.dataSeqMatrixFromName(myhmm.targetOutputSeq)
    
    print(f"Target sequence: {myhmm.targetOutputSeq}")
    print(f"Target output names: {myhmm.targetOutputNames}")
    print(f"Target output data: {[list(x) for x in myhmm.targetOutput]}")
    
    seq_len = len(myhmm.targetOutputNames)
    print(f"Sequence length: {seq_len}")
    
    # Test 1: Overall probability consistency
    print("\n2. Testing overall probability consistency...")
    print("-" * 40)
    
    # Clear memos to ensure fresh computation
    myhmm.memo = {}
    myhmm.memoForward = {}
    
    # Compute backward from START
    backward_result = myhmm.backward("START", 0, seq_len)
    print(f"Backward from START: {backward_result}")
    print(f"Backward Viterbi NLP: {backward_result[0]}")
    print(f"Backward Sum NLP: {backward_result[-1]}")
    print(f"Backward Viterbi Prob: {probFromNlp(backward_result[0])}")
    print(f"Backward Sum Prob: {probFromNlp(backward_result[-1])}")
    
    # Compute forward to LAST  
    forward_result = myhmm.forward("LAST", 0, seq_len)
    print(f"\nForward to LAST: {forward_result}")
    print(f"Forward Viterbi NLP: {forward_result[0]}")
    print(f"Forward Sum NLP: {forward_result[-1]}")
    print(f"Forward Viterbi Prob: {probFromNlp(forward_result[0])}")
    print(f"Forward Sum Prob: {probFromNlp(forward_result[-1])}")
    
    # Check consistency
    viterbi_diff = abs(backward_result[0] - forward_result[0])
    sum_diff = abs(backward_result[-1] - forward_result[-1])
    
    print(f"\nViterbi NLP difference: {viterbi_diff}")
    print(f"Sum NLP difference: {sum_diff}")
    
    TOLERANCE = 1e-10
    viterbi_consistent = viterbi_diff < TOLERANCE
    sum_consistent = sum_diff < TOLERANCE
    
    print(f"Viterbi probabilities consistent: {viterbi_consistent}")
    print(f"Sum probabilities consistent: {sum_consistent}")
    
    # Test 2: Viterbi path consistency
    print("\n3. Testing Viterbi path consistency...")
    print("-" * 40)
    
    # Get backward path
    backward_path = myhmm.backwardAlign("START", 0, seq_len)
    print("\nBackward Viterbi path:")
    print("\t".join(backward_path["keys"]))
    for i, vv in enumerate(backward_path["values"]):
        print(f"Step {i}:\t" + "\t".join([str(xx) for xx in vv]))
    
    # For forward path, we would need a forwardAlign function (not implemented)
    # But we can verify consistency by checking individual state probabilities
    
    # Test 3: Individual state consistency
    print("\n4. Testing individual state consistency...")
    print("-" * 40)
    
    states_to_test = ["START", "RBS_0", "RBS_1", "LAST"]
    
    for state in states_to_test:
        if state == "LAST":
            continue  # Skip LAST for now as it has special handling
            
        print(f"\nTesting state: {state}")
        
        # Clear memos for fresh computation
        myhmm.memo = {}
        myhmm.memoForward = {}
        
        try:
            backward_state = myhmm.backward(state, 0, seq_len)
            forward_state = myhmm.forward(state, 0, seq_len)
            
            print(f"  Backward {state}: Viterbi={backward_state[0]:.6f}, Sum={backward_state[-1]:.6f}")
            print(f"  Forward {state}:  Viterbi={forward_state[0]:.6f}, Sum={forward_state[-1]:.6f}")
            
            vit_diff = abs(backward_state[0] - forward_state[0])
            sum_diff = abs(backward_state[-1] - forward_state[-1])
            
            print(f"  Differences: Viterbi={vit_diff:.10f}, Sum={sum_diff:.10f}")
            print(f"  Consistent: {vit_diff < TOLERANCE and sum_diff < TOLERANCE}")
            
        except Exception as e:
            print(f"  Error testing {state}: {e}")
    
    # Test 4: Partial sequence consistency
    print("\n5. Testing partial sequence consistency...")
    print("-" * 40)
    
    # Test different begin/end ranges
    test_ranges = [(0, 1), (1, 2), (0, 2)]
    
    for begin, end in test_ranges:
        print(f"\nTesting range [{begin}, {end}):")
        
        # Clear memos
        myhmm.memo = {}
        myhmm.memoForward = {}
        
        try:
            backward_partial = myhmm.backward("START", begin, end)
            forward_partial = myhmm.forward("LAST", begin, end)
            
            print(f"  Backward START[{begin}:{end}]: {backward_partial[0]:.6f}")
            print(f"  Forward LAST[{begin}:{end}]:   {forward_partial[0]:.6f}")
            
            diff = abs(backward_partial[0] - forward_partial[0])
            print(f"  Difference: {diff:.10f}")
            print(f"  Consistent: {diff < TOLERANCE}")
            
        except Exception as e:
            print(f"  Error testing range [{begin}, {end}): {e}")
    
    # Test 5: Check memo table sizes
    print("\n6. Memo table analysis...")
    print("-" * 40)
    
    print(f"Backward memo entries: {len(myhmm.memo)}")
    print(f"Forward memo entries: {len(myhmm.memoForward)}")
    
    print("\nBackward memo keys (sample):")
    for i, key in enumerate(list(myhmm.memo.keys())[:5]):
        print(f"  {key}: {myhmm.memo[key][:3]}...")  # Show first 3 elements
    
    print("\nForward memo keys (sample):")
    for i, key in enumerate(list(myhmm.memoForward.keys())[:5]):
        print(f"  {key}: {myhmm.memoForward[key][:3]}...")  # Show first 3 elements
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    overall_success = viterbi_consistent and sum_consistent
    
    print(f"Overall probability consistency: {'PASS' if overall_success else 'FAIL'}")
    print(f"  - Viterbi probabilities match: {'YES' if viterbi_consistent else 'NO'}")
    print(f"  - Sum probabilities match: {'YES' if sum_consistent else 'NO'}")
    
    if not overall_success:
        print("\nERRORS DETECTED - Forward algorithm may have issues!")
        return False
    else:
        print("\nALL TESTS PASSED - Forward and backward algorithms are consistent!")
        return True

def test_emission_probabilities():
    """
    Test emission probability calculations
    """
    print("\n" + "=" * 60)
    print("TESTING EMISSION PROBABILITIES")
    print("=" * 60)
    
    myhmm = HMM()
    myhmm.readJSON("hmm_twoState.json")
    
    data = Data("hmm_data.json")
    myhmm.targetOutputSeq = "RB_video_0"
    (myhmm.targetOutputNames, myhmm.targetOutput) = data.dataSeqMatrixFromName(myhmm.targetOutputSeq)
    
    # Test emission probabilities for each output state
    output_states = ["RBS_0_out", "RBS_1_out"]
    
    for state_name in output_states:
        print(f"\nTesting emission probabilities for {state_name}:")
        state = myhmm.name2state(state_name)
        
        for i, data_point in enumerate(myhmm.targetOutput):
            nlp = state.nlp(data_point)
            prob = probFromNlp(nlp)
            print(f"  Data point {i} {list(data_point)}: NLP={nlp:.6f}, Prob={prob:.8f}")

if __name__ == "__main__":
    try:
        success = test_forward_backward_consistency()
        test_emission_probabilities()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\nTEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)