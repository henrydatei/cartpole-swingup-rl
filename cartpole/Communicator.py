import serial

class Communicator():
    def __init__(self,port,baudrate):
        self.ser = serial.Serial(port, baudrate)
        self.ser.flushInput()
        self.ser.flushOutput()
        self.startChar = b"<"
        self.endChar = b">"

    def send_message(self,message_type:str,angle1:float):
        """Send message via serial and read the receiving result. If the sent and received bytes do not match, it returns False
        """
        if message_type not in ["m", "s", "a", "h", "v", "p", "r"]:
            print("MESSAGE TYPE NOT DEFINED")

        msg_body = message_type.encode() + b',' + str(round(angle1,6)).encode()
        msg = self.startChar + msg_body + self.endChar
        
        self.ser.write(msg)

        list_bytes = []
        while True:
            if self.ser.in_waiting: list_bytes.append(self.ser.read()) #read a single byte if available
            if len(list_bytes) >= len(msg): #wait at least the same amount of bytes as the sended msg are received
                if b"".join(list_bytes[-len(msg):]) == msg: break #stop when the last n bytes are the same as the msg
        
        if len(list_bytes) > len(msg):  payload = (b"".join(list_bytes[:len(list_bytes)-len(msg)])).decode()# in case of message_type "p" some data are send before mirroring the initial msg
        else: payload = None
        
        return True, payload
    
    def close(self):
        self.ser.close()