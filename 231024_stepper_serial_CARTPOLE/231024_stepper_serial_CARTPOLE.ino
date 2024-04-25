// Receive with start- and end-markers combined with parsing

#include "communication_LM.h"
#include "stepper_LM.h"

//============

void setup() {
    Serial.begin(115200);
    Serial.println("____________________________________________________________________________");
    Serial.println("PROJECT CartPole");
    Serial.println("AUTHOR Lars Muschalski (2024)");
    Serial.println();
    Serial.println("Moves stepper with a desired speed.");
    Serial.println("For moving steppers with a desired speed:");
    Serial.println("    <v,speed>");
    Serial.println("For changing the acceleration:");
    Serial.println("    <a,acceleration>");    
    Serial.println();
    Serial.println("Maximum precision is 12 digits: <s,123.456789012>");
    Serial.println();
    Serial.println("For homing / determing the zero point of the steppers  send:");
    Serial.println("    <h,0>");
    Serial.println();
    Serial.println("WARNINGS:");    
    Serial.println("Don't change the acceleration while moving, otherwise the motor might unexpectedly slow down.");
    Serial.println("Don't give a speed greater than defined in setMaxSpeed -> eternal loop.");
    Serial.println("____________________________________________________________________________");
    setup_stepper();
}

//============
void loop() {
    recvWithStartEndMarkers();
    if (newData == true) { 
        strcpy(tempChars, receivedChars);
            // this temporary copy is necessary to protect the original data
            // because strtok() replaces the commas with \0
        parseData(); 

        // Compare messageType to a specific string
        if (strcmp(messageType, "v") == 0) { 
          velocity_setting();
        } else if (strcmp(messageType, "m") == 0) { 
          move_stepper();
        } else if (strcmp(messageType, "a") == 0) { 
          acceleration_setting();
        } else if (strcmp(messageType, "s") == 0) { 
          step_setting();
        } else if (strcmp(messageType, "h") == 0) {
          home_stepper(); //sets the current position as home
        } else if (strcmp(messageType, "p") == 0) {
          get_position(); // get the current position
        } else if (strcmp(messageType, "r") == 0) {
          move_stepper_relative(); // move stepper relative to the current position
        } else {
          //Serial.println("Message doesn't match any known type");
        }
                    
        returnParsedData(); // will be checked by python to verify that the command was correctly received and fully executed
        //returnParsedDataHuman(); // only for debugging and not in conjunction with python!
        newData = false;
    }
    else {
      runSpeedSave();
    }
}
