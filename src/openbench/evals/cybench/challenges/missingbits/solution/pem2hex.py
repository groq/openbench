#!/bin/python3

import base64
import re

from Crypto.Util.asn1 import DerSequence
from Crypto.Util.asn1 import DerObjectId


# DER
DATATYPE = "02"
LENGTH = "8"

ASN1_HEADER_INT = re.compile(DATATYPE + LENGTH)


def parse_value(decoded_lines):
    header_start_posisitons = [
        pos.start()
        for pos in ASN1_HEADER_INT.finditer(decoded_lines)
    ]

    for header_start_posisiton in header_start_posisitons:
        # After 028 the next 4 bit determin the number of bytes
        num_bytes_position = header_start_posisiton + 3
        num_bytes = int(decoded_lines[num_bytes_position], 16)

        # now we can parse the size using the previously gathered information
        size_start = header_start_posisiton + 4
        size_end = header_start_posisiton + 4 + (num_bytes * 2)

        print(decoded_lines[size_start: size_end])
        # Parse the size of the
        parsed_size = int(str(decoded_lines[size_start: size_end]), 16)

        parsed_content = decoded_lines[
            size_end: size_end + (parsed_size * 2)
        ]

        print(parsed_content)


def main() -> int:
    with open("../2048_key.priv", "r") as key_file:
        filecontent_lines = key_file.readlines()

    decoded_lines = ""
    for line in filecontent_lines:
        if line.startswith("-----"):
            continue
        if line.strip() == "":
            print("?"*96)
        else:
            decoded_line = base64.b64decode(line).hex()
            print(decoded_line)
            decoded_lines += decoded_line

    # parse_value(decoded_lines)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
