def get_me1_prep():
    """
    Sub-step 5: Personal Synthesis and Interview Questions
    """
    prep = {
        'Topic': 'Attention Mechanisms in RNNs vs Transformers',
        'Explanation': (
            "During the first 8 weeks, Attention stood out as the most paradigm-shifting but complex topic. "
            "In traditional RNNs, the network compresses an entire sequence into a single fixed-size hidden vector. This creates an information bottleneck, "
            "where early sequence details get 'forgotten' or washed out. Attention solves this by allowing the model to look back at *all* past hidden states "
            "at every decoding step. It calculates a weighted sum of these past states based on how 'relevant' they are to the current prediction. "
            "This essentially provides a dynamic shortcut to exactly where the needed information lives."
        ),
        'Questions': [
            {
                'Q': 'How does self-attention differ from the standard attention mechanism used in seq2seq RNNs?',
                'A': 'Standard seq2seq attention aligns decoded steps to encoded source steps. Self-attention aligns a sequence with itself, calculating the relationships between all words in the exact same sequence to build robust interdependent representations without recurrent steps.'
            },
            {
                'Q': 'What is the vanishing gradient problem and how does an LSTM mitigate it compared to a standard RNN?',
                'A': 'Vanishing gradients occur when multiplying many small derivatives over long sequences, stopping early layers from learning. LSTMs mitigate this with a cell state (a direct additive pathway) that allows gradients to flow backwards uninterrupted through time steps.'
            }
        ]
    }
    return prep
