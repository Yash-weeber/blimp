import cv2
import socket
import struct
import numpy as np


def receive_image(server_ip="0.0.0.0", server_port=5000):
    """
    Set up a TCP server to receive the JPEG image, decode it and then display it.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((server_ip, server_port))
    s.listen(1)
    print("Receiver listening on port", server_port)
    conn, addr = s.accept()
    print("Connected by", addr)

    # Receive a 4-byte header indicating the image data size.
    data_len_bytes = conn.recv(4)
    if len(data_len_bytes) < 4:
        print("Did not receive correct header.")
        return
    data_len = struct.unpack(">I", data_len_bytes)[0]
    data = b""
    while len(data) < data_len:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    img_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is not None:
        cv2.imshow("Received Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to decode received image.")

    conn.close()
    s.close()


if __name__ == "__main__":
    receive_image()
