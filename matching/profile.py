import re
import string


t = string.maketrans('0123456789', '9999999999')


def street_address(address_text, other=None):
    pattern_profile = {
                       '[0-9]+\s+\w+': 'STNUM',
                       '[0-9]+\-[A-Z0-9]+\s+\w+': 'HSTNUM',
                       'P\.*O\.* [Bb][Oo][Xx]\s+[0-9]+': 'POBOX'
                       }
    
    for patt, prof in pattern_profile.items():
        if re.match(patt, address_text):
            return prof
        
    return other if other else address_text
