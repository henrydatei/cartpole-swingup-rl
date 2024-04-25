// source: https://forum.arduino.cc/t/serial-input-basics/278284/72

const byte numChars = 32;
char receivedChars[numChars];
char tempChars[numChars];        // temporary array for use by strtok() function

// variables to hold the parsed data
char messageType[numChars] = {0};
float message = 0;
char startMarker = '<';
char endMarker = '>';
boolean newData = false;
boolean readyForNextCommand = true;
//============


//============

void recvWithStartEndMarkers() {
    static boolean recvInProgress = false;
    static byte ndx = 0;

    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

//============

void parseData() {

      // split the data into its parts
    char * strtokIndx; // this is used by strtok() as an index

    strtokIndx = strtok(tempChars,",");      // get the first part - the string
    strcpy(messageType, strtokIndx); // copy it to messageType
    
    strtokIndx = strtok(NULL, ","); // this continues where the previous call left off
    message = atof(strtokIndx);     // convert this part to a float

}

//============

void returnParsedData() {
  Serial.write(startMarker);
  Serial.write(receivedChars);
  Serial.write(endMarker);
}

void returnParsedDataHuman() {
    Serial.println(receivedChars);
}
