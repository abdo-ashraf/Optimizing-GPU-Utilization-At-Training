import argparse
import torch
import pandas as pd
import time
from Utils.utils import setup_data, setup_model

def flash_attention(number_of_steps:int, batch_size:int):
    # Enable FlashAttention and Memory-Efficient Attention
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=True)
    
    print("Running training with FlashAttention, torch.compile, and BF16 optimization...")
    print(f"Flash Attention Enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Memory Efficient Attention Enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math Attention Enabled: {torch.backends.cuda.math_sdp_enabled()}")
    
    # Setup data and model
    data_x, data_y = setup_data(batch_size=batch_size, num_batch=number_of_steps)
    model = setup_model()
    model.to('cuda')
    
    # Apply torch.compile optimization
    model = torch.compile(model, mode="reduce-overhead")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = pd.read_csv('./results.csv')
    
    # Training loop with FlashAttention, torch.compile, and BF16
    dts = []
    for i in range(number_of_steps):
        t1 = time.time()
        x = data_x[i].to('cuda')
        y = data_y[i].to('cuda')

        optimizer.zero_grad()
        
        # BF16 autocast with FlashAttention
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y, 0)
            
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        dt = (t2 - t1)*1000
        dts.append(round(dt, 2))
        print(f"step {i}, loss: {loss.item():.2f}, dt: {dt:.2f}ms")
    results["BF16+TC+FA"] = dts
    print(f"Logits type: {logits.type()}")
    
    print(f"Average step time (excluding compilation steps): {sum(dts[3:])/len(dts[3:]):.2f}ms")
    
    # Save results to CSV
    results.to_csv("results.csv", index=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run flash_attention')
    parser.add_argument('--number_of_steps', type=int, required=True, 
                        help='Number of training steps to run')
    parser.add_argument('--batch_size', type=int, required=True, 
                        help='Size of training batch')
    
    args = parser.parse_args()
    flash_attention(args.number_of_steps, args.batch_size)


if __name__ == "__main__":
    main()