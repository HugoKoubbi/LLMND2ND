import regex as re
from collections import Counter
import sys
from typing import List, Tuple
print(sys.executable)
print(sys.version)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#for i in range(1, 400):
#    print(chr(i))
# Some tests for the different Utf 
#test_string = "hello! こんにちは!"
#@utf8_encoded = test_string.encode("utf-8")
#print(utf8_encoded)
#print(type(utf8_encoded))
#print(len(test_string))
#print(len(utf8_encoded))
#print(utf8_encoded.decode("utf-8"))
#test_string = "hello! こんにちは!"
#utf16_encoded = test_string.encode("utf-16")
#print(utf16_encoded)
#@print(type(utf16_encoded))
#print(len(test_string))
#print(len(utf16_encoded))
#print(utf16_encoded.decode("utf-16"))
#utf32_encoded = test_string.encode("utf-32")
#print(utf32_encoded)
#print(type(utf32_encoded))
#print(len(test_string))
#print(len(utf32_encoded))
#print(utf32_encoded.decode("utf-32"))
### The UTF 8 seems to have the least number of bytes for representing the same string which is better for attention cost.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
# looking for a counter examples for the above function
# This function is incorrect because it assumes that each byte can be decoded individually,
# which is not the case for UTF-8 encoding. UTF-8 can use multiple bytes to represent a single character,
# especially for characters outside the ASCII range. This function will fail for characters that require more than one byte in UTF-8 encoding.
#print(decode_utf8_bytes_to_str_wrong(b'\xe2\x82\xac'))  # This should raise an error or produce incorrect output
test_string = 'L école de la vie. '
utf8_encode = test_string.encode("utf-8")
print(ord('\xc3'))
print(ord('\xa9'))
print(chr(195))
print(chr(169))
print(utf8_encode)
#print(decode_utf8_bytes_to_str_wrong(utf8_encode))
# The main thing is that the UTF 8 encode some particular characters in two bytes, so an one by one decoding will not work


vocab ={i: (i,) for i in range(256)}
inv_vocab = {v: k for k, v in vocab.items()}
#print(vocab)
a='u'.encode('utf-8')
#print(a)
a = tuple(a)  # Convert bytes to a tuple of bytes
#print(a)
#for i in range(256):
#    vocab[i] = (i,)  # bytes as singleton tuples
#if a in inv_vocab:
#    print("u is in L")
#else:
#    print("u is not in L")

#utf8_encode = test_string.encode("utf-8")
#print(utf8_encode)
#print(type(ord('\xc3')))
#all_bytes_object = bytes(range(256))
#L='miamiamai ejaiehr)&éiréà '.encode('utf-8')
#print(L)
#L = list(L)  # This will give you a list of bytes, each byte is an integer in the range 0-255
#print(L)
# how to get a tuple of bytes from a list of bytes
#L_tuple = tuple(L)  # Convert the list of bytes to a tuple of bytes
#print(L_tuple)
#text='test'
#byte_seq = text.encode('utf-8')
#print(list(byte_seq))  # This will give you a list of bytes
#print([(b,) for b in byte_seq])
text="bla bla bla bo bo bo"
vocab = {i: (i,) for i in range(256)}
inv_vocab = {v: k for k, v in vocab.items()}
print(list(text.encode('utf-8')))
matches = re.finditer(PAT, text)
words = [m.group(0) for m in matches]
sequence_2 = [word.encode('utf-8') for word in words]
seq_counts= Counter(sequence_2)
for seq, count in seq_counts.items():
                for i in range(len(seq) - 1):
                    print(seq[i], seq[i+1])
u=text.encode('utf-8')
v=list(u)
print(u)
print(v)
print(type(u))
print(inv_vocab)

print(seq_counts)
print(sequence_2)
sequences = [tuple(word.encode('utf-8')) for word in words]
print(type(sequences))
sequence_counts = Counter(sequences)
print(type(sequence_counts))
print(sequence_counts)
# Convert each byte to a tuple of byte
text='hdjajbfjbakjbzaf zaknf aknf zaknfbkzfbfkjabfkajzfbazkjfbzakba'
C=Counter(text)
print(max(C, key=C.get))         