import sys
import os
sys.path.append(os.getcwd())
from core.srt_parser import SRTParser

text = "1\n00:00:00,000 --> 00:00:03,500\nHola chicas, mirad que he pasado lo que acabo de recibir\n\n2\n00:00:03,500 --> 00:00:06,340\nel Foreo Luna 4\n\n3\n00:00:06,340 --> 00:00:09,300\nel Foreo Bird 2\n\n4\n00:00:09,300 --> 00:00:13,520\nel Foreo Bird 2 para ojos y labios"

segments = SRTParser.parse_srt(text)
print(f"Found {len(segments)} segments")
for s in segments:
    print(s)
