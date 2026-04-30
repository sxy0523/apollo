#!/usr/bin/env python3
###############################################################################
# Copyright 2026 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

"""Reader for CenterPoint proposal export binary files."""

import argparse
import json
import struct


HEADER = struct.Struct("<8sIIQdII")
RECORD = struct.Struct("<ffffffffi")
FIELDS = ("x", "y", "z", "length", "width", "height", "yaw", "score",
          "class_id")


def read_proposal_export(path):
    with open(path, "rb") as reader:
        raw_header = reader.read(HEADER.size)
        magic, version, record_size, sequence_id, timestamp, count, feature_dim = (
            HEADER.unpack(raw_header))
        if magic != b"APCPROP1":
            raise ValueError("unexpected proposal export magic")
        if version != 1 or record_size != RECORD.size or feature_dim != len(FIELDS):
            raise ValueError("unsupported proposal export format")

        records = []
        for _ in range(count):
            values = RECORD.unpack(reader.read(RECORD.size))
            records.append(dict(zip(FIELDS, values)))
    return {
        "sequence_id": sequence_id,
        "timestamp": timestamp,
        "proposal_count": count,
        "feature_dim": feature_dim,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(read_proposal_export(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
