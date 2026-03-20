import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from loguru import logger
from scapy.all import IP, TCP, Ether, Packet, raw, wrpcap
from scapy.layers.inet import UDP
from scapy.utils import RawPcapReader
from tqdm import tqdm


def stream_pcap(pcap_path):
    """Generator to yield parsed packets and their timestamps from a pcap file"""
    for pkt_data, pkt_metadata in RawPcapReader(pcap_path):
        pkt = Ether(pkt_data)
        ts = pkt_metadata.sec + pkt_metadata.usec / 1e6
        yield pkt, ts


def is_dnp3_packet(pkt):
    # Check if packet has TCP or UDP layer
    if pkt.haslayer(TCP):
        l4 = pkt[TCP]
    elif pkt.haslayer(UDP):
        l4 = pkt[UDP]
    else:
        return False

    # Check if either src or dst port is 20000 (DNP3 default port)
    if l4.sport != 20000 and l4.dport != 20000:
        return False

    # Check if payload exists and starts with DNP3 header bytes 0x05 0x64
    raw = bytes(l4.payload)
    if len(raw) >= 2 and raw[0] == 0x05 and raw[1] == 0x64:
        return True
    return False


@dataclass
class Session:
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
        out_dir: Path = Path("dnp3_labelled_sessions"),
        anynomize: bool = True,
        max_sessions: int = -1,
        correlation_sec: float = 0.574,
        write_every: int = 100,
    ):
        self.out_dir = out_dir
        self.packet_buffer = None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = []
        self.stats = None
        self.anynomize = anynomize
        self.max_sessions = max_sessions
        self.correction_sec = correlation_sec
        self.write_every = write_every
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
        self.packet_buffer = list(stream_pcap(str(self.pcap_path)))
        logger.info(f"Loaded {len(self.packet_buffer)} packets from {self.pcap_path}")

    def packets_to_labelled_sessions(
        self,
        packet_buffer: list[tuple[Packet, float]],
        df: pd.DataFrame = pd.DataFrame(),
    ):
        labelled_sessions = []
        file_path = self.pcap_path
        all_packets = packet_buffer
        pkt_times = [pkt[1] for pkt in all_packets]

        output_path = self.out_dir
        pkt_idxs = list(range(len(all_packets)))

        pbar = tqdm(total=len(df), desc="Packets2Session", unit="session")
        for sess_idx, row in df.iterrows():
            if sess_idx > self.max_sessions and self.max_sessions > 0:
                break
            pbar.update(1)
            start_dt = row["timestamp"] - pd.Timedelta(hours=3)
            end_dt = start_dt + pd.Timedelta(
                microseconds=df.iloc[0].duration, seconds=self.correction_sec
            )
            # convert to seconds
            start_sec = start_dt.timestamp()
            end_sec = end_dt.timestamp()
            total_fwd_pkts = row["TotalFwdPkts"]
            total_bwd_pkts = row["TotalBwdPkts"]
            labled_pkts = total_bwd_pkts + total_fwd_pkts
            matched_idxs = [
                idx for idx in pkt_idxs if start_sec <= pkt_times[idx] <= end_sec
            ]
            idx_matched_pkts = [all_packets[idx] for idx in matched_idxs]

            src_ip = row["source IP"]
            dst_ip = row["destination IP"]
            src_port = row["source port"]
            dst_port = row["destination port"]
            protocol = row["protocol"]
            flow_label = row["Label"]

            matched_pkts = []
            first_pkt = None
            final_matched_idxs = []
            if idx_matched_pkts:
                for idx, pkt in zip(matched_idxs, idx_matched_pkts):
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
                        elif is_dnp3_packet(pkt):
                            is_pkt_matched = True
                        if is_pkt_matched:
                            if self.anynomize:
                                pkt = anonymize_packet(pkt)
                            matched_pkts.append(pkt)
                            final_matched_idxs.append(idx)
                            if not first_pkt:
                                first_pkt = pkt
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
                "flow_id": row["flow ID"],
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "start_time": start_dt,
                "end_time": end_dt,
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
            for idx in final_matched_idxs:
                pkt_idxs.remove(idx)

            labelled_sessions.append(
                Session(
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

        i = 0
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
            i += 1
        logger.info(f"Saved {i} session images to {self.out_dir}")

    def run(self):
        """Run the feature extraction and session processing."""
        # Load packets from PCAP file
        if self.packet_buffer is None:
            self.load()
        else:
            logger.info("Packets already loaded. Skipping load step.")
        if not self.packet_buffer:
            logger.error("No packets loaded. Exiting.")
            return
        logger.info("Starting feature extraction...")
        self.sessions = self.packets_to_labelled_sessions(
            self.packet_buffer, self.label_df
        )
        logger.info(f"Extracted {len(self.sessions)} sessions successfully.")
        # Save session images
        self.sessions_to_image(self.sessions)
        logger.info("Feature extraction completed successfully.")


if __name__ == "__main__":
    timeout = 120
    pcap_root = Path(r"rosaid\data\DNP3 PCAP Files")
    csv_root = (
        Path(r"rosaid\data\Custom_DNP3_Parser")
        / f"{timeout}_timeout"
    )
    out_dir = (
        Path(r"rosaid\data")
        / f"{timeout}_timeout_dnp3_sessions"
    )
    map_file = (
        r"rosaid\assets\dnp3_mapping.json"
    )
    extractor = PCAPSessionFeatureExtractor(out_dir=out_dir, write_every=100)

    with open(map_file, "r") as f:
        dnp3_mapping = json.load(f)

    non_labelled_csv = out_dir.glob("*.csv")
    pcap_files = list(pcap_root.glob("*.pcap"))
    # sort by size
    pcap_files.sort(key=lambda x: x.stat().st_size, reverse=False)
    logger.info(f"Found {len(pcap_files)} pcap files to process.")
    for idx, pcap_file in enumerate(pcap_files):
        csv_name = dnp3_mapping.get(pcap_file.name)
        if not csv_name:
            print(f"No mapping found for {pcap_file.name}")
            continue
        if timeout != 120:
            csv_file = csv_root / csv_name.replace(".pcap", f".pcap{timeout}")
        else:
            csv_file = csv_root / csv_name

        if not csv_file.exists():
            print(f"CSV file not found for {pcap_file.name}")
            continue
        # if "enumerate" not in str(pcap_file).lower():
        #     continue
        df = pd.read_csv(csv_file)
        df.columns = [
            c.strip() for c in df.columns
        ]  # strip whitespace from column names
        # Convert ' date' column to utc datetime and then to timestamp in seconds
        df["timestamp"] = pd.to_datetime(df["date"], dayfirst=True)
        df = df.sort_values(by="timestamp", ascending=True)
        extractor.load(pcap_path=pcap_file, label_df=df)
        extractor.run()
        logger.info(f"Completed {idx + 1}/{len(pcap_files)}")
