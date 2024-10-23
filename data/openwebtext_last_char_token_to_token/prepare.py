import os
import torch
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 8
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    train_portion = 0.1

    full_train = dataset["train"]
    train_size = int(len(full_train) * train_portion)
    
    split_dataset = full_train.train_test_split(
        train_size=train_size,
        test_size=0.0005,
        seed=2357,
        shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        # Prepare the input string
        full_example = "|START|" + example["text"]
        
        # Encode the string to get token IDs
        ids = np.array(enc.encode_ordinary(full_example))
        
        # Get the byte lengths of each token
        token_bytes = [enc.decode_bytes([idx]) for idx in ids]
        token_lengths = np.array([len(tok) for tok in token_bytes])

        num_bytes = np.sum(token_lengths)
        
        # Calculate token start positions
        token_starts = np.cumsum(token_lengths[:-1]) - 1

        input_tokens = np.full(num_bytes, 65535, dtype=np.uint16)
        input_tokens[token_starts] = ids[:-1]
        
        # Create output array
        output_tokens = np.full(num_bytes, -1, dtype=np.int32)
        output_tokens[token_starts] = ids[1:]

        assert len(input_tokens) == len(output_tokens)
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'len': len(input_tokens)
        }

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits, char_to_token ",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(BASE_DIR, f'{split}.bin')
        dtype = np.uint16
        in_arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        out_arr = np.memmap(filename.replace('.bin', '_token_out.bin'), dtype=np.int32, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            in_arr_batch = np.concatenate(batch['input_tokens'])
            out_arr_batch = np.concatenate(batch['output_tokens'])
            in_arr[idx : idx + len(in_arr_batch)] = in_arr_batch
            out_arr[idx : idx + len(out_arr_batch)] = out_arr_batch
            idx += len(in_arr_batch)
        in_arr.flush()
        out_arr.flush()

    print(f"Data processing complete. All files are stored in {BASE_DIR}")