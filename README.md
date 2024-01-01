# AI Simplified Speech Recognition (SSR) Viterbi Algorithm

## Overview
This project presents a simplified approach to speech recognition. It leverages phoneme and fragment mapping to decipher spoken words. The core algorithm utilizes the Viterbi Algorithm for resolving ambiguities in speech recognition.

### Key Features
- **Phoneme-Fragment Mapping**: Direct mapping of phonemes to text fragments for each word, as illustrated with examples like 'water', 'human', and 'ocean'.
- **POMDP-based Approach**: The model treats the process as a Partially Observable Markov Decision Process (POMDP), with text fragments as states and phonemes as observations.
- **Dataset Utilization**: Uses a dataset derived from around 300k Wikipedia articles to compute probabilities.
- **North American English Dialect**: Adopts the CMU Pronouncing Dictionary standards.

## Dataset and Probability Tables
The dataset includes mappings and weights for:
- Fragment to Phoneme pairs
- Fragment to Fragment transitions
- State probabilities and transitions

Probability tables are constructed for:
- Initial State Probability
- State Transition Probability
- Appearance Probabilities

Normalization of weights into appropriate probability distributions is a key part of the process.

## Files in the Repository
- `ssr_viterbi_algorithm.py`: Core Python script for the algorithm.
- `/test_cases`: Folder containing two test cases.
  - Little Prince Environment Test case
  - Simplified Speech Recognition Environment Test case

### Test Case Format
Each test case comprises files for state weights, state-action-state weights, state-observation weights, and observation actions.

## Input/Output Format
The algorithm processes input files containing weights and outputs the most probable sequence of fragments.

### Example Input and Output
**Input**: Sequence of (observation, action) pairs.
**Output**: Predicted state sequence file.

## Environments and Constraints
- **Simplified Speech Recognition Environment**: Caters to 682 states, 69 observations, and a 1-minute time limit per test case.
