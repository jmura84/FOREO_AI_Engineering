"""
SRT subtitle parser and formatter.
Handles parsing, translation, and reconstruction of SRT subtitle files.
"""

import re
from typing import List, Tuple

class SRTParser:
    """Parse and reconstruct SRT subtitle files."""
    
    @staticmethod
    def is_srt_format(text: str) -> bool:
        """Detect if text is in SRT subtitle format.
        
        SRT format:
        1
        00:00:00,000 --> 00:00:03,540
        Subtitle text here
        
        2
        00:00:03,540 --> 00:00:06,380
        More subtitle text
        """
        # Check for SRT pattern: number followed by timestamp
        pattern = r'^\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}'
        return bool(re.search(pattern, text.strip(), re.MULTILINE))
    
    @staticmethod
    def parse_srt(text: str) -> List[Tuple[int, str, List[str]]]:
        """Parse SRT text into structured segments.
        
        Returns:
            List of tuples: (segment_number, timestamp_line, text_lines)
        """
        segments = []
        lines = text.strip().split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Try to parse segment number
            if line.isdigit():
                segment_num = int(line)
                i += 1
                
                # Get timestamp line
                if i < len(lines):
                    timestamp = lines[i].strip()
                    i += 1
                    
                    # Collect subtitle text lines until empty line or next segment
                    text_lines = []
                    while i < len(lines) and lines[i].strip():
                        # Check if it's the start of a new segment
                        if lines[i].strip().isdigit() and i + 1 < len(lines) and '-->' in lines[i + 1]:
                            break
                        text_lines.append(lines[i])
                        i += 1
                    
                    segments.append((segment_num, timestamp, text_lines))
            else:
                i += 1
        
        return segments
    
    @staticmethod
    def reconstruct_srt(segments: List[Tuple[int, str, List[str]]]) -> str:
        """Reconstruct SRT text from parsed segments.
        
        Args:
            segments: List of (segment_number, timestamp_line, text_lines)
            
        Returns:
            Properly formatted SRT text
        """
        result = []
        for segment_num, timestamp, text_lines in segments:
            result.append(str(segment_num))
            result.append(timestamp)
            result.extend(text_lines)
            result.append('')  # Empty line between segments
        
        return '\n'.join(result)
