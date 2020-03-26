import os
import time
import threading
import socket
PORT = 12321
SERVER = "127.0.0.1"
CLIENT = "127.0.0.1"

class Mess(object):
    """The message transfered between client and server"""
    # some order of Message
    # SYN, ACK
    SYN = "SYN"
    ACK = "ACK"

    def __init__(self, src_ip, dst_ip, order=SYN, startT=time.time(), endT=None):
        self.src_ip = src_ip    # source ip
        self.dst_ip = dst_ip    # destination ip
        self.order = order      # mess order
        self.startT = startT    # start time
        self.endT = endT        # end time

    def __str__(self):
        arr = [self.src_ip,
                       self.dst_ip,
                       self.order,
                       self.startT,
                       self.endT]
        arr = [str(x) for x in arr]
        ss = ','.join(arr)
        return ss

    @classmethod
    def from_str(cls, s):
        arr = s.split(',')
        (src_ip, dst_ip, order, startT, endT) = arr
        if startT == 'None':
            startT = None
        else:
            startT = float(startT)
        if endT == 'None':
            endT = None
        else:
            endT = float(endT)
        return Mess(src_ip, dst_ip, order=order, startT=startT, endT=endT)


class Client(object):
    """A simple client to send the mess to server"""
    def __init__(self, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((SERVER, PORT))
        while True:
            mess = Mess(CLIENT, SERVER)
            self.sock.sendall(mess.__str__().encode('utf-8'))
            data = self.sock.recv(1024)
            ack_mess = Mess.from_str(data.decode('utf-8'))
            print(ack_mess)
            print("start time: ", ack_mess.startT)
            print("end time: ", ack_mess.endT)
            print("RTT: ", ack_mess.endT - ack_mess.endT)
            break


class Server(object):
    """A simple server to answer the request from the client"""
    def __init__(self, port=PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((SERVER, port))
        self.start_serve()

    def start_serve(self):
        while True:
            data, address = self.sock.recvfrom(1024)
            mess = Mess.from_str(data.decode('utf-8'))
            mess.src_ip, mess.dst_ip = mess.dst_ip, mess.src_ip
            mess.order = Mess.ACK
            mess.endT = time.time()
            # newdata = mess.__str__().encode('utf-8')
            self.sock.sendto(mess.__str__().encode('utf-8'), address)

if __name__ == "__main__":
    server = Server()
    # client = Client(12456)