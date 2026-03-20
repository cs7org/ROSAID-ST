import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from scapy.all import IP, TCP, Ether, Packet, raw, rdpcap, wrpcap
from scapy.layers.inet import UDP
from scapy.utils import RawPcapReader
from tqdm import tqdm

# root dir
parser = argparse.ArgumentParser(description="PCAP to Session Image Converter")
parser.add_argument(
    "--work_dir",
    type=str,
    default=r"rosaid",
    help="Working directory for logs and outputs",
)

parser.add_argument(
    "--data_dir",
    type=str,
    default=r"data",
    help="Root directory containing PCAP files and CSV labels",
)

args = parser.parse_args()


class PacketStreamer:
    def __init__(
        self, pcap_path: Path, store_packets: bool = False, use_editcap: bool = True
    ):
        self.pcap_path = pcap_path
        self.curr_index = 0
        self.curr_packet = None
        self.curr_packet_time = None
        self.store_packets = store_packets
        self.use_editcap = use_editcap
        self.all_packets = []

    def __iter__(self):
        """Generator to yield parsed packets and their timestamps from a pcap file"""
        for pkt_data, pkt_metadata in RawPcapReader(str(self.pcap_path)):
            pkt = Ether(pkt_data)
            ts = pkt_metadata.sec + pkt_metadata.usec / 1e6
            self.curr_index += 1
            if self.store_packets:
                self.all_packets.append((pkt, ts))
            yield pkt, ts

    def split_session(self, start_ts: float, end_ts: float):
        """Split packets into sessions based on start and end timestamps using editcap"""
        import subprocess
        from datetime import datetime, timedelta

        # Convert epoch to datetime
        start_dt = datetime.fromtimestamp(float(start_ts))
        end_dt = datetime.fromtimestamp(float(end_ts))

        # Add 2-second tolerance (common fix for microsecond mismatches)
        tolerance = timedelta(seconds=2)
        start_dt -= tolerance
        end_dt += tolerance

        start_formatted = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_formatted = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Writing packets of window: {start_formatted} → {end_formatted}")

        # Split with tolerance
        cmd = f'editcap "{self.pcap_path}" "temp.pcap" -A "{start_formatted}" -B "{end_formatted}"'

        # Execute command and wait for it to complete
        process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"editcap failed: {process.stderr}")
        logger.info("Completed splitting PCAP with editcap.")
        return None

    def get_packets(
        self,
        num_packets: int | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ):
        logger.info(f"Retrieving packets from {self.pcap_path}")
        packets = []

        if not self.use_editcap:
            if num_packets is not None:
                for _ in range(num_packets):
                    try:
                        pkt, ts = next(self.__iter__())
                        packets.append((pkt, ts))
                    except StopIteration:
                        break
            elif start_ts is not None and end_ts is not None:
                for pkt_ts in self.__iter__():
                    pkt, ts = pkt_ts
                    if start_ts <= ts <= end_ts:
                        packets.append((pkt, ts))
                    elif ts > end_ts:
                        break
                logger.info(
                    f"Retrieved {len(packets)} packets from PCAP between {start_ts} and {end_ts}."
                )
        else:
            # Use editcap to split pcap and read packets
            self.split_session(start_ts, end_ts)
            all_packets = rdpcap("temp.pcap")
            packets = []
            for pkt in all_packets:
                ts = pkt.time

                packets.append((pkt, ts))

            logger.info(
                f"Retrieved {len(packets)} packets from PCAP between {start_ts} and {end_ts} using editcap."
            )
        return packets


@dataclass
class Session:
    index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    packets: list[Packet]
    interval: float
    raw_bytes: list[bytearray]
    filename: str | None = None
    array: np.ndarray = None
    label: str = "NORMAL"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_packets(self) -> int:
        return len(self.packets)

    def __repr__(self):
        return (
            f"Session(start_time={self.start_time}, end_time={self.end_time}, "
            f"num_packets={self.num_packets}, interval={self.interval})"
        )


def anonymize_packet(packet: Packet) -> Packet:
    """Anonymize packet by removing address information"""
    # Create copy to avoid modifying original packet
    pkt = packet.copy()

    # IP layer handling
    if pkt.haslayer(IP):
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"

    # Ethernet layer handling
    if pkt.haslayer(Ether):
        pkt[Ether].src = "00:00:00:00:00:00"
        pkt[Ether].dst = "00:00:00:00:00:00"

    # TCP layer handling
    if pkt.haslayer(TCP):
        pkt[TCP].sport = 0
        pkt[TCP].dport = 0

    return pkt


class PCAPSessionFeatureExtractor:
    def __init__(
        self,
        out_dir: Path = Path("iec104_labelled_sessions"),
        anynomize: bool = True,
        max_sessions: int = -1,
        correction_msec: float = 0.0,
        write_every: int = 100,
        min_labeled_pkts: int = -1,
        max_labeled_pkts: int = -1,
        adaptive_correction_msec: bool = True,
    ):
        self.out_dir = out_dir
        self.packet_buffer = None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = []
        self.stats = None
        self.anynomize = anynomize
        self.max_sessions = max_sessions
        self.correction_msec = correction_msec
        self.write_every = write_every
        self.min_labeled_pkts = min_labeled_pkts
        self.max_labeled_pkts = max_labeled_pkts
        self.adaptive_correction_msec = adaptive_correction_msec
        with open(out_dir / "labelled_sessions.csv", "w") as f:
            # f.write(
            #     "flow_id,src_ip,dst_ip,src_port,dst_port,protocol,start_time,end_time,total_matched_pkts,total_labeled_pkts,matched_forward_pkts,matched_backward_pkts,labled_forward_pkts,labled_backward_pkts,session_file_name,flow_label\n"
            # )
            pass

    def load(self, pcap_path: Path, label_df: pd.DataFrame):
        """Load packets from the PCAP file."""
        self.pcap_path = pcap_path
        self.label_df = label_df
        logger.info(f"Loading packets from {self.pcap_path}...")
        self.packet_streamer = PacketStreamer(self.pcap_path)
        logger.info("Created PacketStreamer for packet loading.")

    def packets_to_labelled_sessions(
        self,
        packet_streamer: PacketStreamer,
        df: pd.DataFrame = pd.DataFrame(),
    ):
        labelled_sessions = []
        file_path = self.pcap_path

        output_path = self.out_dir

        df["total_pkts"] = df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"]
        num_rows = len(df)
        if self.min_labeled_pkts > 0:
            logger.info(
                f"Filtering sessions with min_labeled_pkts={self.min_labeled_pkts}"
            )
            num_rows = len(df)
            df = df.query("total_pkts >= @self.min_labeled_pkts")
            logger.info(f"Filtered {num_rows - len(df)} sessions")
        if self.max_labeled_pkts > 0:
            df = df.query("total_pkts <= @self.max_labeled_pkts")
            logger.info(f"Filtered {num_rows - len(df)} sessions")

        if self.adaptive_correction_msec:
            logger.warning("Adaptive correction is not implemented yet.")

        last_packet_time = None
        pbar = tqdm(total=len(df), desc="Packets2Session", unit="session")
        for sess_idx, row in df.iterrows():
            if sess_idx > self.max_sessions and self.max_sessions > 0:
                break
            pbar.update(1)
            # do everything in microseconds
            # else there will be precision issues!!
            start_dt = row["timestamp"]
            start_ts = start_dt.timestamp() * 1e6
            end_ts = start_ts + row["Flow Duration"] + self.correction_msec
            # convert back to datetime
            end_dt = pd.to_datetime(end_ts, unit="us")
            # convert to seconds
            start_sec = start_dt.timestamp()
            end_sec = end_dt.timestamp()
            total_fwd_pkts = row["Tot Fwd Pkts"]
            total_bwd_pkts = row["Tot Bwd Pkts"]

            labled_pkts = total_bwd_pkts + total_fwd_pkts
            found_packets = packet_streamer.get_packets(
                start_ts=start_sec, end_ts=end_sec
            )
            # if last_packet_time is None:
            #     # get (packets, ts) in the time range
            #     found_packets = packet_streamer.get_packets(
            #         start_ts=start_sec, end_ts=end_sec
            #     )
            #     last_packet_time = end_sec
            # else:
            #     # if last packet time is before start_sec, get packets from last_packet_time to end_sec
            #     if end_sec > last_packet_time:
            #         found_packets = packet_streamer.get_packets(
            #             start_ts=last_packet_time, end_ts=end_sec
            #         )
            #         last_packet_time = end_sec

            # matched_idxs = [
            #     idx for idx in pkt_idxs if start_sec <= pkt_times[idx] <= end_sec
            # ]
            # idx_matched_pkts = [all_packets[idx] for idx in matched_idxs]

            src_ip = row["Src IP"]
            dst_ip = row["Dst IP"]
            src_port = row["Src Port"]
            dst_port = row["Dst Port"]
            protocol = row["Protocol"]
            flow_label = row["Label"]

            matched_pkts = []
            first_pkt = None
            matched_pkt_idxs = []
            if found_packets:
                for pkt_idx, (pkt, ts) in enumerate(found_packets):
                    # break if we have enough packets
                    # bcz new pkts could be in different flow
                    if len(matched_pkts) >= labled_pkts:
                        break
                    pkt = pkt[0]  # Extract the packet from the tuple
                    if not (pkt.haslayer(IP) and pkt.haslayer(Ether)):
                        continue
                    ip_layer = pkt.getlayer(IP) or pkt.getlayer("IPv6")
                    if not ip_layer:
                        continue
                    if (ip_layer.src == src_ip and ip_layer.dst == dst_ip) or (
                        ip_layer.src == dst_ip and ip_layer.dst == src_ip
                    ):
                        is_pkt_matched = False
                        if pkt.haslayer(TCP):
                            if (
                                pkt[TCP].sport == src_port
                                and pkt[TCP].dport == dst_port
                            ) or (
                                pkt[TCP].sport == dst_port
                                and pkt[TCP].dport == src_port
                            ):
                                is_pkt_matched = True
                        elif pkt.haslayer(UDP):
                            if (
                                pkt.getlayer(UDP).sport == src_port
                                and pkt.getlayer(UDP).dport == dst_port
                            ) or (
                                pkt.getlayer(UDP).sport == dst_port
                                and pkt.getlayer(UDP).dport == src_port
                            ):
                                is_pkt_matched = True

                        if is_pkt_matched:
                            if self.anynomize:
                                pkt = anonymize_packet(pkt)
                            matched_pkts.append(pkt)
                            if not first_pkt:
                                first_pkt = pkt
                            matched_pkt_idxs.append(pkt_idx)
                pbar.set_postfix(
                    dict(
                        matched_pkts=len(matched_pkts),
                        total_pkts=labled_pkts,
                        flow_label=flow_label,
                    )
                )
            num_forward_pkts = len(
                [pkt for pkt in matched_pkts if pkt.src == first_pkt.src]
            )
            raw_bytes = [raw(pkt) for pkt in matched_pkts]
            raw_lengths = [len(byt) for byt in raw_bytes]
            max_length = max(raw_lengths) if raw_lengths else 0
            min_length = min(raw_lengths) if raw_lengths else 0
            avg_length = sum(raw_lengths) / len(raw_lengths) if raw_lengths else 0
            num_backward_pkts = len(matched_pkts) - num_forward_pkts
            part = self.pcap_path.stem.split(".")[0]
            session_file_name = f"{flow_label}_{sess_idx}_{part}.pcap"
            labelled_session = {
                "flow_id": row["Flow ID"],
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "start_time": start_dt,
                "end_time": end_dt,
                "start_timestamp": start_sec,
                "end_timestamp": end_sec,
                "total_matched_pkts": len(matched_pkts),
                "total_labeled_pkts": labled_pkts,
                "matched_forward_pkts": num_forward_pkts,
                "matched_backward_pkts": num_backward_pkts,
                "labled_forward_pkts": total_fwd_pkts,
                "labled_backward_pkts": total_bwd_pkts,
                "raw_bytes_max_length": max_length,
                "raw_bytes_min_length": min_length,
                "raw_bytes_avg_length": avg_length,
                "session_file_name": session_file_name,
                "flow_label": flow_label,
                "input_file": file_path.name,
            }

            # save session info to csv
            with open(output_path / "labelled_sessions.csv", "a") as f:
                keys = labelled_session.keys()
                if f.tell() == 0:  # write header if file is empty
                    f.write(",".join(keys) + "\n")
                # write session info
                f.write(",".join([str(labelled_session[key]) for key in keys]) + "\n")

            # save session packets to a pcap file
            session_pcap_path = output_path / "session_pcaps"
            if not session_pcap_path.exists():
                session_pcap_path.mkdir(parents=True, exist_ok=True)
            session_pcap_path = session_pcap_path / session_file_name
            if not matched_pkts:
                continue
            wrpcap(str(session_pcap_path), matched_pkts)

            # remove matched packets from found_packets
            for idx in sorted(matched_pkt_idxs, reverse=True):
                del found_packets[idx]

            labelled_sessions.append(
                Session(
                    index=sess_idx,
                    filename=session_file_name,
                    start_time=start_dt,
                    end_time=end_dt,
                    packets=matched_pkts,
                    interval=end_dt - start_dt,
                    raw_bytes=raw_bytes,
                    label=flow_label,
                )
            )
            if len(labelled_sessions) >= self.write_every:
                self.sessions_to_image(labelled_sessions)
                labelled_sessions = []
        return labelled_sessions

    def extract_session_features(self, session_bytes):
        """
        Extract first N bytes from first M packets of a session

        Args:
            session_bytes (list): List of pkt bytes session

        Returns:
            tuple: (8x128 grayscale array, 8x128 byte sequence array)
        """
        max_packets = len(session_bytes)
        bytes_per_packet = max([len(byt) for byt in session_bytes])
        # Initialize arrays
        grayscale_data = np.zeros((max_packets, bytes_per_packet), dtype=np.uint8)

        # Process up to max_packets
        processed_packets = 0
        for i, packet in enumerate(session_bytes):
            try:
                packet = packet
                raw_bytes = raw(packet)

                # Extract first bytes_per_packet bytes
                packet_data = raw_bytes[:bytes_per_packet]

                # Pad if necessary
                if len(packet_data) < bytes_per_packet:
                    packet_data += b"\x00" * (bytes_per_packet - len(packet_data))

                # Convert to numpy array
                packet_array = np.frombuffer(packet_data, dtype=np.uint8)

                # Store in both formats
                grayscale_data[processed_packets] = packet_array

                processed_packets += 1

            except Exception as e:
                print(f"Error processing packet {i}: {e}")
                continue

        return grayscale_data

    def normalized_features(self, packets: list[Packet]):
        """
        Based on: ByteStack‑ID: Integrated Stacked Model Leveraging Payload Byte Frequency for Grayscale Image‑based Network Intrusion Detection
        NOTE: frequency distribution-based packet-level **PAYLOAD** to image generation
        """
        num_pkts = len(packets)
        image = np.zeros((num_pkts, 256), dtype=np.float32)

        for i, pkt in enumerate(packets):
            # Extract payload bytes (handles different packet representations)
            if hasattr(pkt, "load"):
                raw_bytes = bytes(pkt.load) if pkt.load else b""
            elif isinstance(pkt, bytes):
                raw_bytes = pkt
            else:
                raw_bytes = b""

            if not raw_bytes:
                # Empty payload results in zero vector
                continue

            # Calculate byte frequency distribution
            byte_counts = np.zeros(256, dtype=np.float32)
            for byte_val in raw_bytes:
                byte_counts[byte_val] += 1

            # Packet-specific normalization (as per ByteStack-ID)
            max_freq = byte_counts.max()
            if max_freq > 0:
                byte_counts /= max_freq

            image[i, :] = byte_counts
        image = (image * 255).astype(np.uint8)
        return image

    def extract_sessions(self):
        if not self.packet_buffer:
            logger.warning("No packets found in the PCAP file.")
            return
        logger.info(f"Extracting sessions from {len(self.packet_buffer)} packets.")

        logger.info(f"Processing interval: {self.interval} seconds")
        self.sesessions = self.packets_to_labelled_sessions(
            self.packet_buffer, df=self.label_df
        )

        return self.sessions

    def sessions_to_image(self, sessions: list[Session]):
        """Convert sessions to grayscale images and save them."""
        if not sessions:
            return
        # Use sequential processing for small batches
        for session in tqdm(sessions, desc="Session2Image", unit="session"):
            if not session.packets:
                continue
            img_name = session.filename.replace(".pcap", ".png")
            image_dir = self.out_dir / "session_images" / img_name
            if not image_dir.parent.exists():
                image_dir.parent.mkdir(parents=True)

            # Extract features
            grayscale_array = self.extract_session_features(session.raw_bytes)
            normalized_array = self.normalized_features(session.packets)
            cv2.imwrite(str(image_dir), grayscale_array)
            cv2.imwrite(
                str(image_dir).replace(".png", "_normalized.png"), normalized_array
            )
        logger.info(f"Saved {len(sessions)} session images to {self.out_dir}")

    def run(self):
        """Run the feature extraction and session processing."""
        # Load packets from PCAP file
        if self.packet_streamer is None:
            self.load()
        else:
            logger.info("Packets already loaded. Skipping load step.")
        if not self.packet_streamer:
            logger.error("No packets loaded. Exiting.")
            return
        logger.info("Starting feature extraction...")
        self.sessions = self.packets_to_labelled_sessions(
            self.packet_streamer, self.label_df
        )
        logger.info(f"Extracted {len(self.sessions)} sessions successfully.")
        # Save remaining session images
        self.sessions_to_image(self.sessions)
        logger.info("Feature extraction completed successfully.")


if __name__ == "__main__":
    import json

    timeout = 120
    pcap_root = Path(args.data_dir)
    out_dir = pcap_root.parent / f"{pcap_root.name}_sessions"
    work_dir = Path(args.work_dir)

    completed_attacks = []
    extractor = PCAPSessionFeatureExtractor(out_dir=out_dir, write_every=20)
    pcap_files = list(pcap_root.glob("*.pcap"))
    mapping_path = work_dir / "assets" / "rosid23_pcap_csv_mapping.json"

    # read mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    # sort by size
    pcap_files.sort(key=lambda x: x.stat().st_size, reverse=False)
    for idx, pcap_file in enumerate(pcap_files):
        logger.info(f"Processing {pcap_file.name}...")
        # average(percentile(num_packets, 90)) of all
        max_labeled_pkts = 4378
        min_labeled_pkts = -1

        extractor.min_labeled_pkts = min_labeled_pkts
        extractor.max_labeled_pkts = max_labeled_pkts

        atk_name = pcap_file.stem.split("-")[-1]
        if atk_name in completed_attacks:
            logger.info(
                f"Skipping {pcap_file.name} as attack {atk_name} in completed attacks"
            )
            continue
        csv_name = mapping[pcap_file.stem] + ".csv"
        csv_file = pcap_file.parent / csv_name

        if not csv_file.exists():
            logger.warning(f"CSV file not found for {pcap_file.name}")
            continue
        df = pd.read_csv(csv_file)
        df.columns = [
            c.strip() for c in df.columns
        ]  # strip whitespace from column names

        df["timestamp"] = pd.to_datetime(df["Timestamp"])
        # subtract 3 hours to match pcap timing
        df["timestamp"] = df["timestamp"] - pd.Timedelta(hours=3)
        df = df.sort_values(by="timestamp", ascending=True)
        # df = df.sort_values(by="Flow Duration", ascending=True)

        df.Label = df.Label.apply(lambda x: atk_name if int(x) == 1 else "NORMAL")

        extractor.load(pcap_path=pcap_file, label_df=df)
        extractor.run()
        logger.info(f"Completed {idx}/{len(pcap_files)}")
