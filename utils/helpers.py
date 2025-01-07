import matplotlib.pyplot as plt

def plot_phone_ratios(phone_vocabs, train_ratio, valid_ratio, test_ratio, output_path=None):
    plt.figure(figsize=(25,10))
    plt.bar(phone_vocabs, train_ratio, label='Train')
    plt.bar(phone_vocabs, valid_ratio, bottom=train_ratio, label='Validation')
    bottom_combined = [x + y for x, y in zip(train_ratio, valid_ratio)]
    plt.bar(phone_vocabs, test_ratio, bottom=bottom_combined, label='Test')
    plt.xlabel('Phonemes')
    plt.ylabel('Ratio')
    plt.title('Phoneme Distribution')
    plt.legend()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
