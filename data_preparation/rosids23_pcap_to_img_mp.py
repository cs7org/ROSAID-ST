import argparse
import gc
import multiprocessing
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
from loguru import logger
from scapy.all import IP, TCP, Ether, Packet, raw, rdpcap, wrpcap
from scapy.layers.inet import UDP
from scapy.utils import RawPcapReader
from tqdm import tqdm

# root dir
parser = argparse.ArgumentParser(description="PCAP to Session Image Converter")
parser.add_argument(
    "--project_dir",
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
parser.add_argument(
    "--out_dir",
    type=str,
    default=r"rosaid\data\rosids23_sessions",
    help="Root directory for output images and sessions",
)
parser.add_argument(
    "--temp_dir",
    type=str,
    default=None,
    help="Temporary directory for intermediate files",
)
parser.add_argument(
    "--num_processes",
    type=int,
    default=6,
    help="Number of processes to use for processing",
)
parser.add_argument(
    "--compress_to",
    type=str,
    default=None,
    help="Directory to zip and move processed files to",
)
parser.add_argument(
    "--use_apptainer",
    action="store_true",
    default=False,
    help="Use Apptainer for containerized processing",
)
parser.add_argument(
    "--use_tshark",
    action="store_true",
    default=True,
    help="Use Tshark for packet processing",
)
parser.add_argument(
    "--container",
    type=str,
    default="docker://cincan/tshark",
    help="Container image to use with Apptainer",
)

args = parser.parse_args()
log_path = Path(args.project_dir) / "logs" / "rosid23_pcap_to_img_mp.log"

# log file in writing mode
logger.add(
    log_path,
)


class PacketStreamer:
    def __init__(
        self,
        pcap_path: Path,
        name: str | None = None,
        temp_dir: Path | None = None,
        store_packets: bool = False,
        use_editcap: bool = True,
        use_tshark: bool = True,
        process_id: int = 0,
        use_apptainer=True,
        container: str = "docker://cincan/tshark",
        start_timestamp: float | None = None,
        end_timestamp: float | None = None,
    ):
        self.use_apptainer = use_apptainer
        self.container = container
        self.pcap_path = pcap_path
        self.curr_index = 0
        self.curr_packet = None
        self.curr_packet_time = None
        self.store_packets = store_packets
        self.use_editcap = use_editcap
        self.use_tshark = use_tshark
        self.all_packets = []
        # if available, use temp dir else create temp in working dir
        self.temp_dir = temp_dir if temp_dir else Path.cwd() / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # if available, else random name
        self.temp_name = name if name else f"process_{process_id}"
        self.temp_path = self.temp_dir / f"{self.temp_name}_temp.pcap"
        self.process_id = process_id
        self.first_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

        if self.use_editcap:
            if self.first_timestamp and self.end_timestamp:
                logger.info(
                    f"PROCESS:{self.process_id} PacketStreamer initializing for timestamps between {self.first_timestamp} and {self.end_timestamp}"
                )
                split_path = self.temp_dir / f"{self.temp_name}_split.pcap"
                # now split the pcap to only include packets between these timestamps
                ret_code = self.split_session(
                    self.first_timestamp,
                    self.end_timestamp,
                    initial_split=True,
                    split_path=split_path,
                )
                if not ret_code:
                    logger.error(
                        f"PROCESS:{self.process_id} Failed to initialize PacketStreamer with editcap splitting."
                    )
                    raise RuntimeError("Failed to split pcap with editcap.")
                logger.info(
                    f"PROCESS:{self.process_id} Successfully initialized PacketStreamer with editcap splitting and new split file at {split_path}."
                )
                self.pcap_path = split_path

    def __iter__(self):
        """Generator to yield parsed packets and their timestamps from a pcap file"""
        for pkt_data, pkt_metadata in RawPcapReader(str(self.pcap_path)):
            pkt = Ether(pkt_data)
            ts = pkt_metadata.sec + pkt_metadata.usec / 1e6
            self.curr_index += 1
            if self.store_packets:
                self.all_packets.append((pkt, ts))
            yield pkt, ts

    def split_session(
        self,
        start_ts: float,
        end_ts: float,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        src_port: int | None = None,
        dst_port: int | None = None,
        initial_split: bool = False,
        split_path: Path | None = None,
    ):
        """Split packets into sessions based on start and end timestamps using editcap or scapy fallback"""
        if split_path is None:
            split_path = self.temp_path
        # Convert epoch to datetime
        start_dt = datetime.fromtimestamp(float(start_ts))
        end_dt = datetime.fromtimestamp(float(end_ts))

        # Add 1-second tolerance (common fix for microsecond mismatches)
        tolerance = timedelta(seconds=1)
        start_dt -= tolerance
        end_dt += tolerance

        start_formatted = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_formatted = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            f"PROCESS:{self.process_id} Writing packets of window: {start_formatted} → {end_formatted}"
        )

        # Try editcap first bcz its fast
        # change split_path
        if self.use_apptainer:
            editcap_cmd = [
                "apptainer",
                "exec",
                "--no-home",
                "--cleanenv",
                self.container,
                "editcap",
                str(self.pcap_path),
                str(split_path),
                "-A",
                start_formatted,
                "-B",
                end_formatted,
            ]
        else:
            editcap_cmd = shlex.split(
                f'editcap "{self.pcap_path}" "{split_path}" -A "{start_formatted}" -B "{end_formatted}"'
            )
        filtered_packets = None
        try:
            # Execute command and wait for it to complete
            # also print full error if any
            t0 = time.time()
            process = subprocess.run(
                editcap_cmd,
                capture_output=True,  # FIXED: NO shell=True for lists
                text=True,
            )
            t1 = time.time()

            if process.returncode != 0:
                logger.error(f"EDITCAP FAILED: {process.stderr}")
                logger.error(f"EDITCAP CMD: {' '.join(map(str, editcap_cmd))}")
                raise subprocess.CalledProcessError(
                    process.returncode, editcap_cmd, process.stderr
                )

            logger.info(
                f"PROCESS:{self.process_id} Completed splitting PCAP ({start_formatted} → {end_formatted}) with editcap. Time taken: {t1 - t0:.2f} seconds"
            )
            pcap_path = self.temp_path
            if initial_split:
                return True
            # NOTE: This is clearly slower than editcap then tshark filtering
            # # now make tshark cmd that filters by src and dst ip if provided aand also implements start and end time
            # # and write the time taken to find best one
            # SRC_IP = src_ip
            # DST_IP = dst_ip
            # tshark_cmd = f"""tshark -r {str(self.pcap_path)} \
            # -Y "frame.time >= \\"{start_formatted}\\" && frame.time <= \\"{end_formatted}\\" && ((ip.src == {SRC_IP} && ip.dst == {DST_IP}) || (ip.src == {DST_IP} && ip.dst == {SRC_IP}))" \
            # -w {self.temp_path}
            # """
            # logger.info(
            #     f"PROCESS:{self.process_id} Running tshark command: {tshark_cmd}"
            # )
            # t0 = time.time()
            # tshark_process = subprocess.run(
            #     tshark_cmd, shell=True, capture_output=True, text=True
            # )
            # t1 = time.time()
            # logger.info(
            #     f"PROCESS:{self.process_id} Completed filtering PCAP with tshark. Time taken: {t1 - t0:.2f} seconds"
            # )
            # if tshark_process.returncode != 0:
            #     raise subprocess.CalledProcessError(
            #         tshark_process.returncode, tshark_cmd, tshark_process.stderr
            #     )

            # use tshark now to filter by src and dst ip if provided
            if (src_ip or dst_ip) and self.use_tshark:
                # either src==src_ip and dst==dst_ip or src==dst_ip and dst==src_ip
                # Replace with your IPs
                SRC_IP = src_ip
                DST_IP = dst_ip
                filtered_path = (
                    self.temp_path.parent / f"{self.temp_name}_filtered.pcap"
                )

                if self.use_apptainer:
                    tshark_cmd = f"""apptainer exec --no-home --cleanenv {self.container} tshark -r {str(self.temp_path)} """
                    tshark_cmd += f"""-Y "(ip.src == {SRC_IP} && ip.dst == {DST_IP}) || (ip.src == {DST_IP} && ip.dst == {SRC_IP})" """
                    if src_port is not None and dst_port is not None:
                        tshark_cmd += f"""-Y "((tcp.srcport == {src_port} && tcp.dstport == {dst_port}) || (tcp.srcport == {dst_port} && tcp.dstport == {src_port}))" """
                    tshark_cmd += f"""-w {str(filtered_path)}"""
                else:
                    tshark_cmd = f"""tshark -r {str(self.temp_path)} -Y "(ip.src == {SRC_IP} && ip.dst == {DST_IP}) || (ip.src == {DST_IP} && ip.dst == {SRC_IP})" """
                    if src_port is not None and dst_port is not None:
                        tshark_cmd += f"""-Y "((tcp.srcport == {src_port} && tcp.dstport == {dst_port}) || (tcp.srcport == {dst_port} && tcp.dstport == {src_port}))" """
                    tshark_cmd += f"""-w {filtered_path}"""
                # logger.info(
                #     f"PROCESS:{self.process_id} Running tshark command: {tshark_cmd}"
                # )
                t0 = time.time()
                tshark_process = subprocess.run(
                    tshark_cmd, capture_output=True, text=True, shell=True
                )
                t1 = time.time()
                logger.info(
                    f"PROCESS:{self.process_id} Completed filtering PCAP with tshark. Time taken: {t1 - t0:.2f} seconds"
                )
                if tshark_process.returncode != 0:
                    logger.error(f"TSHARK FAILED - stderr: {tshark_process.stderr}")
                    logger.error(f"TSHARK CMD: {' '.join(map(str, tshark_cmd))}")
                    logger.error(f"TSHARK STDOUT: {tshark_process.stdout}")
                    raise subprocess.CalledProcessError(
                        tshark_process.returncode, tshark_cmd, tshark_process.stderr
                    )
                # remove the unfiltered temp file
                self.temp_path.unlink()
                pcap_path = filtered_path

            # Read filtered packets from temp file
            # but first check the available memory to avoid crashes
            # and then assume how much memory the pcap would take and if enough, read it
            # else wait for few seconds and retry
            while True:
                gc.collect()
                mem_available = (
                    psutil.virtual_memory().available * 0.7
                )  # 70% of available memory
                pcap_size = pcap_path.stat().st_size

                if mem_available > pcap_size:
                    filtered_packets = rdpcap(str(pcap_path))
                    break
                else:
                    logger.warning(
                        f"PROCESS:{self.process_id} Not enough memory to read {self.temp_path}. Waiting..."
                    )
                    time.sleep(5)
            # remove this temp file
            pcap_path.unlink()

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"EDITCAP/TSHARK ERROR: {e}")
            logger.warning(
                f"PROCESS:{self.process_id} failed: {e}\n {traceback.format_exc()}"
            )
            logger.info(f"PROCESS:{self.process_id} Falling back to Scapy filtering")

            # Scapy fallback - filter packets by timestamp and write to temp file
            filtered_packets = []
            adjusted_start_ts = start_ts - 2.0  # Apply tolerance
            adjusted_end_ts = end_ts + 2.0

            try:
                for pkt, ts in self.__iter__():
                    if adjusted_start_ts <= ts <= adjusted_end_ts:
                        filtered_packets.append(pkt)

                return filtered_packets

            except Exception as scapy_error:
                logger.error(
                    f"PROCESS:{self.process_id} Scapy fallback failed: {scapy_error} \n {traceback.format_exc()}"
                )

            logger.info(
                f"PROCESS:{self.process_id} Completed splitting PCAP ({start_formatted} → {end_formatted}) with Scapy fallback."
            )
        return filtered_packets

    def get_packets(
        self,
        num_packets: int | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        src_ip: str | None = None,
        dst_ip: str | None = None,
        src_port: int | None = None,
        dst_port: int | None = None,
    ):
        logger.info(
            f"PROCESS:{self.process_id} Retrieving packets from {self.pcap_path}"
        )
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
                    f"PROCESS:{self.process_id} Retrieved {len(packets)} packets from PCAP between {start_ts} and {end_ts}."
                )
        else:
            t0 = time.time()
            # Use editcap to split pcap and read packets
            all_packets = self.split_session(
                start_ts,
                end_ts,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
            )

            try:
                packets = []
                for pkt in all_packets:
                    ts = pkt.time
                    packets.append((pkt, ts))

                logger.info(
                    f"PROCESS:{self.process_id} Retrieved {len(packets)} packets from PCAP between {start_ts} and {end_ts} in {time.time() - t0:.2f} seconds."
                )
            except Exception as e:
                logger.error(
                    f"PROCESS:{self.process_id} Error reading temp file {self.temp_path}: {e}\n {traceback.format_exc()}"
                )
                return packets
        return packets

    def cleanup(self):
        """Cleanup temporary files"""
        if self.temp_path.exists():
            self.temp_path.unlink()
            logger.info(
                f"PROCESS:{self.process_id} Deleted temporary file: {self.temp_path}"
            )


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
        process_id: int,
        out_dir: Path = Path("iec104_labelled_sessions"),
        anynomize: bool = True,
        max_sessions: int = -1,
        correction_msec: float = 0.0,
        write_every: int = 100,
        min_labeled_pkts: int = -1,
        max_labeled_pkts: int = -1,
        adaptive_correction_msec: bool = True,
        temp_dir: Path | None = None,
        use_apptainer: bool = True,
        container: str = "docker://cincan/tshark",
        use_tshark: bool = True,
    ):
        self.use_tshark = use_tshark
        self.container = container
        self.use_apptainer = use_apptainer
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
        self.process_id = process_id
        self.temp_dir = temp_dir if temp_dir else Path.cwd() / "temp"
        self.label_df = None
        self.packet_streamer = None
        self.pcap_path = None
        self.label_file = None

    def load(self, pcap_path: Path, label_df: pd.DataFrame):
        """Load packets from the PCAP file."""
        self.pcap_path = pcap_path
        self.label_df = label_df
        logger.info(
            f"PROCESS:{self.process_id} Loading packets from {self.pcap_path}..."
        )
        start_dt = label_df["timestamp"].to_list()[0]
        end_ts = (
            label_df["timestamp"].to_list()[-1].timestamp() * 1e6
            + label_df["Flow Duration"].to_list()[-1]
            + self.correction_msec
        )
        # convert back to datetime
        end_dt = pd.to_datetime(end_ts, unit="us")
        # convert to seconds
        start_sec = start_dt.timestamp()
        end_sec = end_dt.timestamp()
        self.packet_streamer = PacketStreamer(
            self.pcap_path,
            process_id=self.process_id,
            temp_dir=self.temp_dir,
            use_apptainer=self.use_apptainer,
            container=self.container,
            start_timestamp=start_sec,
            end_timestamp=end_sec,
            use_tshark=self.use_tshark,
        )
        logger.info(
            f"PROCESS:{self.process_id} Created PacketStreamer for packet loading."
        )
        # format as yyyyMMdd_HHmm
        now = datetime.now().strftime("%Y%m%d_%H%M")
        self.label_file = (
            self.out_dir
            / f"labelled_sessions_{pcap_path.stem}_{self.process_id}_{now}.csv"
        )
        with open(self.label_file, "w") as f:
            pass

    def packets_to_labelled_sessions(
        self,
        packet_streamer: PacketStreamer,
        df: pd.DataFrame = pd.DataFrame(),
    ):
        labelled_sessions = []
        file_path = self.pcap_path

        output_path = self.out_dir

        num_rows = len(df)
        if self.min_labeled_pkts > 0:
            logger.info(
                f"PROCESS:{self.process_id} Filtering sessions with min_labeled_pkts={self.min_labeled_pkts}"
            )
            num_rows = len(df)
            df = df.query("total_pkts >= @self.min_labeled_pkts")
            logger.info(
                f"PROCESS:{self.process_id} Filtered {num_rows - len(df)} sessions"
            )
        if self.max_labeled_pkts > 0:
            df = df.query("total_pkts <= @self.max_labeled_pkts")
            logger.info(
                f"PROCESS:{self.process_id} Filtered {num_rows - len(df)} sessions"
            )

        if self.adaptive_correction_msec:
            logger.warning(
                f"PROCESS:{self.process_id} Adaptive correction is not implemented yet."
            )

        last_packet_time = None
        pbar = tqdm(
            total=len(df),
            desc=f"Pkts2Sess (PID: {self.process_id})",
            unit="session",
            disable=not sys.stdout.isatty(),
        )
        total_rows = len(df)
        curr_row = 0
        found_packets = []
        for sess_idx, row in df.iterrows():
            curr_row += 1
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
            src_ip = row["Src IP"]
            dst_ip = row["Dst IP"]
            src_port = row["Src Port"]
            dst_port = row["Dst Port"]
            protocol = row["Protocol"]
            flow_label = row["Label"]

            labled_pkts = row["total_pkts"]
            logger.info(
                f"PROCESS:{self.process_id} Processing session {curr_row}/{total_rows}: Flow ID {row['Flow ID']} with {labled_pkts} labeled packets."
            )
            matched_pkts = []
            matched_pkt_idxs = []
            attempt = 1
            # try to reuse found_packets from previous session to avoid re-reading pcap
            while True:
                if attempt > 2 or len(matched_pkts) >= labled_pkts:
                    break
                if not found_packets or len(matched_pkts) < labled_pkts:
                    found_packets = packet_streamer.get_packets(
                        start_ts=start_sec,
                        end_ts=end_sec,
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=src_port,
                        dst_port=dst_port,
                    )

                attempt += 1
                first_pkt = None
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
                    "session_index": sess_idx,
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
                with open(self.label_file, "a") as f:
                    keys = labelled_session.keys()
                    if f.tell() == 0:  # write header if file is empty
                        f.write(",".join(keys) + "\n")
                    # write session info
                    f.write(
                        ",".join([str(labelled_session[key]) for key in keys]) + "\n"
                    )

                # save session packets to a pcap file
                session_pcap_path = output_path / "session_pcaps"
                if not session_pcap_path.exists():
                    session_pcap_path.mkdir(parents=True, exist_ok=True)
                session_pcap_path = session_pcap_path / session_file_name
                if not matched_pkts:
                    continue
                # remove matched packets from found_packets
                found_packets = [
                    pkt
                    for idx, pkt in enumerate(found_packets)
                    if idx not in matched_pkt_idxs
                ]

            wrpcap(str(session_pcap_path), matched_pkts)

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
                # Force garbage collection after batch processing

                gc.collect()

            logger.info(
                f"PROCESS:{self.process_id} Processed session {curr_row}/{total_rows}: {session_file_name} with {len(matched_pkts)}/{labled_pkts} matched packets."
            )
            # Memory cleanup - Clear large variables after processing each session
            del matched_pkts, raw_bytes, raw_lengths
        pbar.close()
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

                # Clean up temporary variables for large packets
                del packet_data, packet_array, raw_bytes

            except Exception as e:
                logger.error(
                    f"PROCESS:{self.process_id} Error processing packet {i}: {e} \n {traceback.format_exc()}"
                )
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

            # Clean up temporary variables for memory management
            del byte_counts, raw_bytes

        image = (image * 255).astype(np.uint8)
        return image

    def extract_sessions(self):
        if not self.packet_buffer:
            logger.warning("No packets found in the PCAP file.")
            return
        logger.info(
            f"PROCESS:{self.process_id} Extracting sessions from {len(self.packet_buffer)} packets."
        )

        logger.info(
            f"PROCESS:{self.process_id} Processing interval: {self.interval} seconds"
        )
        self.sessions = self.packets_to_labelled_sessions(
            self.packet_buffer, df=self.label_df
        )

        return self.sessions

    def sessions_to_image(self, sessions: list[Session]):
        """Convert sessions to grayscale images and save them."""
        if not sessions:
            return
        # Use sequential processing for small batches
        for session in tqdm(
            sessions,
            desc=f"Session2Image (PID: {self.process_id})",
            unit="session",
            disable=not sys.stdout.isatty(),
        ):
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

            # Clean up arrays after saving to free memory
            del grayscale_array, normalized_array

        logger.info(
            f"PROCESS:{self.process_id} Saved {len(sessions)} session images to {self.out_dir}"
        )

        # Force garbage collection after processing batch of sessions

        gc.collect()

    def run(self):
        """Run the feature extraction and session processing."""
        # Load packets from PCAP file
        if self.packet_streamer is None:
            self.load()
        else:
            logger.info(
                f"PROCESS:{self.process_id} Packets already loaded. Skipping load step."
            )
        if not self.packet_streamer:
            logger.error("No packets loaded. Exiting.")
            return
        logger.info(f"PROCESS:{self.process_id} Starting feature extraction...")
        self.sessions = self.packets_to_labelled_sessions(
            self.packet_streamer, self.label_df
        )
        logger.info(
            f"PROCESS:{self.process_id} Extracted {len(self.sessions)} sessions successfully."
        )
        # Save remaining session images
        self.sessions_to_image(self.sessions)
        logger.info(
            f"PROCESS:{self.process_id} Feature extraction completed successfully."
        )


def run_extractor(
    process_id: int,
    pcap_path: Path,
    label_df: pd.DataFrame,
    out_dir: Path,
    min_labeled_pkts: int = -1,
    max_labeled_pkts: int = -1,
    temp_dir: Path | None = None,
    use_apptainer: bool = True,
    container: str = "docker://cincan/tshark",
    use_tshark: bool = True,
):
    extractor = PCAPSessionFeatureExtractor(
        out_dir=out_dir,
        write_every=1,
        min_labeled_pkts=min_labeled_pkts,
        max_labeled_pkts=max_labeled_pkts,
        process_id=process_id,
        adaptive_correction_msec=False,
        temp_dir=temp_dir,
        use_apptainer=use_apptainer,
        container=container,
        use_tshark=use_tshark,
    )
    extractor.load(pcap_path=pcap_path, label_df=label_df)
    extractor.run()
    extractor.packet_streamer.cleanup()


if __name__ == "__main__":
    import json

    # NOTE: this is replaced by 80 pctl of each label's num_packets
    # average(percentile(num_packets, 90)) of all labels
    max_labeled_pkts = 4378
    min_labeled_pkts = -1
    pcap_root = Path(args.data_dir)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pcap_root.parent / f"{pcap_root.name}_sessions"
    project_dir = Path(args.project_dir)
    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = project_dir / "temp"
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
    # unauthorizedsubscriber is written as unauthorizedsubsriber in csv
    completed_attacks = [
        # "unauthorizedpublisher",
        # "unauthorizedsubsriber",
        # "subcriberflood",
    ]

    pcap_files = list(pcap_root.glob("*.pcap"))
    mapping_path = project_dir / "assets" / "rosid23_pcap_csv_mapping.json"

    # true cpu cores but not logical cores
    cpu_cores = max(multiprocessing.cpu_count() - 1, 1)
    if args.num_processes > 0:
        cpu_cores = min(cpu_cores, args.num_processes)
    logger.info(f"Using {cpu_cores} CPU cores.")

    # read mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    # sort by size
    pcap_files.sort(key=lambda x: x.stat().st_size, reverse=False)
    for idx, pcap_file in enumerate(pcap_files):
        logger.info(f"Processing {pcap_file.name}...")

        atk_name = pcap_file.stem.split("-")[-1]

        if atk_name in completed_attacks:
            logger.info(
                f"Skipping {pcap_file.name} as attack {atk_name} in completed attacks"
            )
            continue
        if pcap_file.stem not in mapping:
            logger.warning(
                f"PCAP file {pcap_file.name} not found in mapping. Skipping."
            )
            continue
        csv_name = mapping[pcap_file.stem] + ".csv"
        csv_file = pcap_file.parent / csv_name

        # read completed csv
        completed_csv_files = list(out_dir.rglob(f"*{atk_name}*csv"))
        completed_df = pd.DataFrame()
        for completed_csv in completed_csv_files:
            try:
                temp_df = pd.read_csv(completed_csv)
                completed_df = pd.concat([completed_df, temp_df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error reading completed CSV file ({completed_csv}): {e}")
                # remove corrupted file
                completed_csv.unlink()
        logger.info(
            f"Found {len(completed_df)} completed sessions for attack {atk_name} from {len(completed_csv_files)} files."
        )

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

        logger.info(f"{csv_file} Labels: {df['Label'].value_counts()}")

        def map_label(x):
            if x == "No Label":
                return "NORMAL"
            if int(x) == 1:
                return atk_name
            else:
                return "NORMAL"

        df.Label = df.Label.apply(map_label)
        df["total_pkts"] = df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"]

        # find max_labeled_pkts as 80 percentile
        max_labeled_pkts = df["total_pkts"].quantile(0.80)

        logger.info(f"Total sessions in CSV before filtering: {len(df)}")
        # filter based on min and max labeled pkts
        if min_labeled_pkts > 0:
            df = df.query("total_pkts >= @min_labeled_pkts")
        if max_labeled_pkts > 0:
            df = df.query("total_pkts <= @max_labeled_pkts")
        logger.info(f"Total sessions in CSV after filtering: {len(df)}")

        if not completed_df.empty:
            initial_len = len(df)
            completed_df["session_index"] = completed_df["session_file_name"].apply(
                lambda x: int(x.split("_")[1])
                if isinstance(x, str) and len(x.split("_")) > 1
                else -1
            )
            # make it index
            completed_indices = set(completed_df["session_index"].unique())
            ndf = df[~df.index.isin(completed_indices)]
            logger.info(
                f"Removed {initial_len - len(ndf)} completed sessions from processing."
            )
            df = ndf

        del completed_df

        # # NOTE: Remove this part after done
        # if 'dos' in atk_name.lower():
        #     # only read normal session
        #     df = df[df["Label"] == "NORMAL"]
        #     logger.info(f"Filtered to only NORMAL sessions for DoS attack: {len(df)} sessions.")

        # get num rows
        num_rows = len(df)
        if num_rows == 0:
            logger.info(f"No sessions to process for {pcap_file.name}. Skipping.")
            continue
        num_rows_per_core = num_rows // cpu_cores
        if num_rows_per_core < 1:
            raise ValueError(
                f"Not enough rows ({num_rows}) for the number of CPU cores ({cpu_cores})"
            )

        # last core also takes the remainder
        if num_rows % cpu_cores > 0:
            num_rows_per_core += 1
        logger.info(
            f"Splitting {num_rows} rows into chunks of {num_rows_per_core} for {cpu_cores} cores"
        )

        # split df into chunks
        df_chunks = [
            df.iloc[i : i + num_rows_per_core]
            for i in range(0, num_rows, num_rows_per_core)
        ]
        logger.info(f"Created {len(df_chunks)} chunks for processing.")
        # now process then multiprocessing
        with multiprocessing.Pool(processes=cpu_cores) as pool:
            pool.starmap(
                run_extractor,
                [
                    (
                        process_id + 1,
                        pcap_file,
                        df_chunk,
                        out_dir,
                        min_labeled_pkts,
                        max_labeled_pkts,
                        temp_dir,
                        args.use_apptainer,
                        args.container,
                        args.use_tshark,
                    )
                    for process_id, df_chunk in enumerate(df_chunks)
                ],
            )
            pool.close()
            pool.join()

        # Clean up memory after processing each file
        del df, df_chunks

        gc.collect()

        logger.info(f"Completed {idx}/{len(pcap_files)}")
    # at the end read all the labelled_sessions_*.csv and merge into one
    all_label_files = list(out_dir.glob("labelled_sessions_*.csv"))
    if all_label_files:
        merged_df = pd.concat(
            [pd.read_csv(f) for f in all_label_files], ignore_index=True
        )
        merged_df.to_csv(out_dir / "labelled_sessions.csv", index=False)
        logger.info(f"Merged {len(all_label_files)} labelled session files into one.")

    # Optionally zip and move processed files
    if args.compress_to:
        import shutil

        zip_path = Path(args.compress_to)
        shutil.make_archive(
            base_name=str(zip_path).replace(".zip", ""),
            format="zip",
            root_dir=out_dir,
        )
        logger.info(f"Zipped processed sessions to {zip_path}")
