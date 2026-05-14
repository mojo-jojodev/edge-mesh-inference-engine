import torch
import socket
import struct
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - WORKER - %(message)s')

class EdgeWorker:
    """Executes a partitioned shard of an LLM via TCP socket stream."""
    def __init__(self, port: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.port = port
        self.device = device
        self.layer_weights = torch.randn(4096, 4096, device=self.device) # Mock 7B layer
        
    def process_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """Applies local layer transformations."""
        with torch.no_grad():
            return torch.matmul(activations, self.layer_weights)

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen(5)
            logging.info(f"Node active on port {self.port}, compute device: {self.device}")
            while True:
                conn, addr = s.accept()
                with conn:
                    # Receive 8-byte header for tensor size
                    header = conn.recv(8)
                    if not header: break
                    msg_len = struct.unpack("!Q", header)[0]
                    
                    # Read tensor payload
                    data = bytearray()
                    while len(data) < msg_len:
                        packet = conn.recv(4096)
                        if not packet: break
                        data.extend(packet)
                    
                    import io
                    buffer = io.BytesIO(data)
                    incoming_tensor = torch.load(buffer, map_location=self.device)
                    logging.info(f"Received activation tensor of shape {incoming_tensor.shape}")
                    
                    # Compute and return
                    output_tensor = self.process_activations(incoming_tensor)
                    out_buffer = io.BytesIO()
                    torch.save(output_tensor.cpu(), out_buffer)
                    payload = out_buffer.getvalue()
                    
                    conn.sendall(struct.pack("!Q", len(payload)) + payload)

if __name__ == "__main__":
    worker = EdgeWorker(port=8080)
    # Uncomment to run listener: worker.start_server()
