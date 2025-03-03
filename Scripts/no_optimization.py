from Utils.utils import setup_data, setup_model
import torch
import pandas as pd
import time
import argparse

def no_optimization(number_of_steps:int):
    # Disable any optimizations that might be on by default
    torch.backends.cuda.enable_flash_sdp(enabled=False)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    
    print("Running training with no optimizations...")
    
    # Setup data and model
    data_x, data_y = setup_data()
    model = setup_model()
    model.to('cuda')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = pd.read_csv('./results.csv')
    
    # Training loop with no optimizations
    dts = []
    for i in range(number_of_steps):
        t1 = time.time()
        x = data_x[i].to('cuda')
        y = data_y[i].to('cuda')

        optimizer.zero_grad()
        logits, loss = model(x, y, 0)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t2 = time.time()
        dt = (t2 - t1)*1000
        dts.append(round(dt, 2))
        print(f"step {i}, loss: {loss.item():.2f}, dt: {dt:.2f}ms")
    results["no_optimization"] = dts
    print(f"Logits type: {logits.dtype}")
        
    print(f"Average step time (excluding first compilation step): {sum(dts[1:])/len(dts[1:]):.2f}ms")
    
    # Save results to CSV.
    results.to_csv("results.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Run no-optimization')
    parser.add_argument('--number_of_steps', type=int, required=True, 
                        help='Number of training steps to run')
    
    args = parser.parse_args()
    no_optimization(args.number_of_steps)


if __name__ == "__main__":
    main()