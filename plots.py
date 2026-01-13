import json
import matplotlib.pyplot as plt


def plot_balanced_experiments(small_path, large_path, title, save_name):
    with open(small_path, 'r') as f: data_s = json.load(f)
    with open(large_path, 'r') as f: data_l = json.load(f)
    
    t_loss_s = [x['train']['loss'] for x in data_s]
    v_loss_s = [x['val']['loss'] for x in data_s]
    
    t_loss_l = [x['train']['loss'] for x in data_l]
    v_loss_l = [x['val']['loss'] for x in data_l]
    
    max_len = max(len(t_loss_s), len(t_loss_l))
    steps = range(max_len)
    
    def pad_list(lst, target_len):
        return lst + [None] * (target_len - len(lst))

    t_loss_s = pad_list(t_loss_s, max_len)
    v_loss_s = pad_list(v_loss_s, max_len)
    t_loss_l = pad_list(t_loss_l, max_len)
    v_loss_l = pad_list(v_loss_l, max_len)

    plt.figure(figsize=(12, 7))
    
    plt.plot(steps, t_loss_s, label='Small: Train Loss', color='blue', linestyle='--', alpha=0.6)
    plt.plot(steps, v_loss_s, label='Small: Val Loss', color='blue', linewidth=2)
    
    plt.plot(steps, t_loss_l, label='Large: Train Loss', color='red', linestyle='--', alpha=0.6)
    plt.plot(steps, v_loss_l, label='Large: Val Loss', color='red', linewidth=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Evaluation Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_name)


plot_balanced_experiments('./results/char_small_results.json', './results/char_large_results.json', 'Character-Level Comparison', 'char_loss.png')
plot_balanced_experiments('./results/subword_small_results.json', './results/subword_large_results.json', 'Subword-Level Comparison', 'subword_loss.png')