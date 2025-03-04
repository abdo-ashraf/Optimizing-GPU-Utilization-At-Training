import argparse
import torch
import pandas as pd
import time
from Utils.utils import setup_data, setup_model

def torch_compile(number_of_steps:int):
    # Disable other optimizations
    torch.backends.cuda.enable_flash_sdp(enabled=False)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    
    print("Running training with torch.compile and BF16 optimization...")
    
    # Setup data and model
    data_x, data_y = setup_data(batch_size=256, num_batch=number_of_steps)
    model = setup_model()
    model.to('cuda')
    
    # Apply torch.compile optimization
    # Use "reduce-overhead" mode for best performance
    model = torch.compile(model, mode="reduce-overhead")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = pd.read_csv('./results.csv')
    
    # Training loop with torch.compile and BF16
    dts = []
    for i in range(number_of_steps):
        t1 = time.time()
        x = data_x[i].to('cuda')
        y = data_y[i].to('cuda')

        optimizer.zero_grad()
        
        # BF16 autocast
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y, 0)
            
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        dt = (t2 - t1)*1000
        dts.append(round(dt, 2))
        print(f"step {i}, loss: {loss.item():.2f}, dt: {dt:.2f}ms")
    results["BF16+TC"] = dts
    print(f"Logits type: {logits.type()}")
    
    print(f"Average step time (excluding first compilation step): {sum(dts[1:])/len(dts[1:]):.2f}ms")
    
    # Save results to CSV
    results.to_csv("results.csv", index=False)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run torch_compile')
    parser.add_argument('--number_of_steps', type=int, required=True, 
                        help='Number of training steps to run')
    
    args = parser.parse_args()
    torch_compile(args.number_of_steps)


if __name__ == "__main__":
    main()