import argparse
import torch
import pandas as pd
import time
from Utils.utils import setup_data, setup_model

def tensorFloat32(number_of_steps:int, batch_size:int):
    # Disable other optimizations
    torch.backends.cuda.enable_flash_sdp(enabled=False)
    torch.backends.cuda.enable_mem_efficient_sdp(enabled=False)
    
    # Enable TensorFloat32 (TF32)
    torch.set_float32_matmul_precision('high')  # 'high' corresponds to TF32
    
    print("Running training with TensorFloat32 (TF32) optimization...")
    
    # Setup data and model
    data_x, data_y = setup_data(batch_size=batch_size, num_batch=number_of_steps)
    model = setup_model()
    model.to('cuda')
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = pd.read_csv('./results.csv')
    
    # Training loop with TF32 optimization
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
    results["TF32"] = dts
    print(f"Logits type: {logits.dtype}")
    
    print(f"Average step time (excluding compilation steps): {sum(dts[3:])/len(dts[3:]):.2f}ms")
    
    # Save results to CSV
    results.to_csv("results.csv", index=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run tensorFloat32')
    parser.add_argument('--number_of_steps', type=int, required=True, 
                        help='Number of training steps to run')
    parser.add_argument('--batch_size', type=int, required=True, 
                        help='Size of training batch')
    
    args = parser.parse_args()
    tensorFloat32(args.number_of_steps, args.batch_size)


if __name__ == "__main__":
    main()