import cv2
import numpy as np
from ip import IP
from collections import Counter
import heapq

class Compress(IP):
    def __init__(self, path):
        super().__init__(path)
        self.gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) if len(self.img.shape) == 3 else self.img
        self.data = self.gray_img.flatten().tolist()
    
    def huffman(self):
        if not self.data:
            return "", {}
        
        freq = Counter(self.data)
        
        if len(freq) == 1:
            symbol = list(freq.keys())[0]
            return "0" * len(self.data), {symbol: "0"}
        
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            
            for pair in lo[1:]:
                pair[1] = "0" + pair[1]
            for pair in hi[1:]:
                pair[1] = "1" + pair[1]
            
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        huff_map = {symbol: code for symbol, code in sorted(heap[0][1:])}
        encoded = "".join(huff_map[value] for value in self.data)
        
        return encoded, huff_map
    
    def golomb_rice(self, M=4):
        if M <= 0 or (M & (M - 1)) != 0:
            raise ValueError("M must be a power of 2")
        
        k = int(np.log2(M))
        encoded = []
        
        for x in self.data:
            q = x // M
            r = x % M
            unary = "1" * q + "0"
            binary = format(r, f"0{k}b")
            encoded.append(unary + binary)
        
        return encoded
    
    def arithmetic(self):
        if not self.data:
            return 0.0
        
        freq = Counter(self.data)
        total = len(self.data)
        probs = {k: v / total for k, v in freq.items()}
        
        cum = {}
        cum_val = 0.0
        for symbol in sorted(probs.keys()):
            prob = probs[symbol]
            cum[symbol] = (cum_val, cum_val + prob)
            cum_val += prob
        
        low, high = 0.0, 1.0
        for symbol in self.data:
            range_width = high - low
            high = low + range_width * cum[symbol][1]
            low = low + range_width * cum[symbol][0]
        
        return (low + high) / 2, cum
    
    def lzw(self):
        dictionary_size = 256
        dictionary = {bytes([i]): i for i in range(dictionary_size)}
        
        result = []
        w = bytes()
        
        for value in self.data:
            c = bytes([value])
            wc = w + c
            
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dictionary_size
                dictionary_size += 1
                w = c
        
        if w:
            result.append(dictionary[w])
        
        return result, dictionary_size
    
    def rle(self):
        if not self.data:
            return []
        
        encoded = []
        current_value = self.data[0]
        count = 1
        
        for value in self.data[1:]:
            if value == current_value and count < 255:
                count += 1
            else:
                encoded.append((current_value, count))
                current_value = value
                count = 1
        
        encoded.append((current_value, count))
        return encoded
    
    def symbol_based(self):
        freq = Counter(self.data)
        sorted_symbols = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        
        symbol_map = {symbol: idx for idx, (symbol, _) in enumerate(sorted_symbols)}
        encoded = [symbol_map[value] for value in self.data]
        
        return encoded, symbol_map
    
    def bit_plane(self):
        bit_planes = []
        for bit_position in range(8):
            plane = (self.gray_img >> bit_position) & 1
            bit_planes.append(plane)
        
        return bit_planes
    
    def dct_blocks(self, block_size=8):
        h, w = self.gray_img.shape
        
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        
        padded_img = np.pad(self.gray_img, ((0, pad_h), (0, pad_w)), mode='edge')
        new_h, new_w = padded_img.shape
        
        dct_result = np.zeros((new_h, new_w), dtype=np.float32)
        
        for i in range(0, new_h, block_size):
            for j in range(0, new_w, block_size):
                block = np.float32(padded_img[i:i+block_size, j:j+block_size])
                dct_block = cv2.dct(block)
                dct_result[i:i+block_size, j:j+block_size] = dct_block
        
        return dct_result[:h, :w]
    
    def predictive(self, mode='left'):
        h, w = self.gray_img.shape
        predicted = np.zeros_like(self.gray_img, dtype=np.int16)
        residual = np.zeros_like(self.gray_img, dtype=np.int16)
        
        for i in range(h):
            for j in range(w):
                if mode == 'left':
                    pred_value = self.gray_img[i, j-1] if j > 0 else 128
                elif mode == 'top':
                    pred_value = self.gray_img[i-1, j] if i > 0 else 128
                elif mode == 'avg':
                    left = self.gray_img[i, j-1] if j > 0 else 128
                    top = self.gray_img[i-1, j] if i > 0 else 128
                    pred_value = (int(left) + int(top)) // 2
                else:
                    pred_value = 128
                
                predicted[i, j] = pred_value
                residual[i, j] = int(self.gray_img[i, j]) - pred_value
        
        return residual, predicted
    
    def wavelet(self, level=1):
        img = np.float32(self.gray_img)
        
        for _ in range(level):
            h, w = img.shape
            
            if h < 2 or w < 2:
                break
            
            new_h = h // 2
            new_w = w // 2
            
            result = np.zeros_like(img)
            
            for i in range(new_h):
                for j in range(w):
                    result[i, j] = (img[2*i, j] + img[2*i+1, j]) / 2
                    result[new_h+i, j] = (img[2*i, j] - img[2*i+1, j]) / 2
            
            temp = result.copy()
            for i in range(h):
                for j in range(new_w):
                    result[i, j] = (temp[i, 2*j] + temp[i, 2*j+1]) / 2
                    result[i, new_w+j] = (temp[i, 2*j] - temp[i, 2*j+1]) / 2
            
            img = result
        
        return img
    
    def get_compression_stats(self, encoded_data, original_bits=None):
        if original_bits is None:
            original_bits = len(self.data) * 8
        
        if isinstance(encoded_data, str):
            compressed_bits = len(encoded_data)
        elif isinstance(encoded_data, list):
            compressed_bits = len(encoded_data) * 16
        elif isinstance(encoded_data, np.ndarray):
            compressed_bits = encoded_data.size * 8
        else:
            compressed_bits = original_bits
        
        compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0
        space_saving = ((original_bits - compressed_bits) / original_bits * 100) if original_bits > 0 else 0
        
        return {
            'original_bits': original_bits,
            'compressed_bits': compressed_bits,
            'compression_ratio': round(compression_ratio, 2),
            'space_saving': round(space_saving, 2)
        }