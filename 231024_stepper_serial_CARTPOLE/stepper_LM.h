/*
 * TO BE USED WITH ESP32
 * STEPPER SHOULD BE USED IN XXX STEP
 * 
 *
 * WIRING
 * GPIO   PIN OF...
 * 
 * PUL_LEFT      27
 * DIR_LEFT      26
 * +5V_LEFT      3.3V
 * ENBL_LEFT     25
 * 
 *    
 * ENDSWITCH_LEFT   18
 * 
 * 13     PUL   (left stepper)
 * 14     DIR   (left stepper)
 * 34	    ENABLE (left stepper)
 * 33     ENDSWITCH LEFT 'NO'
 * 3V3    +5V (left stepper), ENDSWITCH 'C'
 * 
 */

#include <AccelStepper.h>

const uint8_t BUILTIN_LED      = 2;
const uint8_t endswitch_left   = 18;
const uint8_t PUL_LEFT         = 27;
const uint8_t DIR_LEFT         = 26;
const uint8_t ENABLE_LEFT      = 25;

int   stepsPerRevolution  = 12800;
int   maxRevolutions = 9; //change this according to how many revolutions are possible without bumping into the rig.


bool speadReached = false;
long speedIncrement;
unsigned long currentTime;
unsigned long previousTime;
float desiredSpeed;
int interval = 10;  // # of milliseconds between speed increases

// Define a stepper and the pins it will use
AccelStepper stepper(AccelStepper::DRIVER,PUL_LEFT,DIR_LEFT); // Defaults to AccelStepper::FULL4WIRE (4 pins) on 2, 3, 4, 5


void setup_stepper(){
  pinMode(BUILTIN_LED, OUTPUT);
  pinMode(endswitch_left, INPUT_PULLUP);
  stepper.setMaxSpeed(stepsPerRevolution*5);
  stepper.setAcceleration(stepsPerRevolution*5);
  stepper.setEnablePin(ENABLE_LEFT);
  // home_stepper();
}

void get_position(){
  Serial.print(stepper.currentPosition());
}

void runSpeedSave(){
  //Serial.print("currentPosition:");
  //Serial.println(stepper.currentPosition());
  if (abs(stepper.currentPosition()) < maxRevolutions*stepsPerRevolution) stepper.runSpeed();
  else if (stepper.currentPosition() >= 0 && stepper.speed()<0) stepper.runSpeed();
  else if (stepper.currentPosition() <= 0 && stepper.speed()>0) stepper.runSpeed();
}

void changeSpeed(){
  currentTime = millis();
  while(speadReached == false){
    previousTime = currentTime;
    while ((currentTime-previousTime)<interval){ //wait until the interval is past
      currentTime = millis();
      runSpeedSave();
    }
    speedIncrement = stepper.acceleration()*(currentTime-previousTime)/1000;
    if (stepper.speed() < desiredSpeed) {   
      stepper.setSpeed(stepper.speed()+speedIncrement);
      if (stepper.speed() > desiredSpeed) { //overshooting of speed
        stepper.setSpeed(desiredSpeed);
        speadReached = true;
      }
    }
    else if (stepper.speed() > desiredSpeed) {
      stepper.setSpeed(stepper.speed()-speedIncrement);
      if (stepper.speed() < desiredSpeed) { //undershooting of speed
        stepper.setSpeed(desiredSpeed);
        speadReached = true;
      }      
    }
    else speadReached = true;
    Serial.print("stepper.currentPosition():");
    Serial.print(stepper.currentPosition());   
    Serial.print(",");
    Serial.print("stepper.speed():");
    Serial.println(stepper.speed());    
    runSpeedSave();
  }
}

void velocity_setting(){
  //stepper.setMaxSpeed(int(message));
  speadReached = false;
  desiredSpeed = message;
  changeSpeed();
}

void acceleration_setting(){
  stepper.setAcceleration(int(message));
}

void step_setting(){
  stepsPerRevolution = int(message);
}

void home_stepper(){
  stepsPerRevolution = int(message);
  stepper.setCurrentPosition(0);
}

// void move_stepper(){
//   stepper.moveTo(message);
//   while (abs(stepper.distanceToGo()) > 0 ) stepper.run();
// }

void move_stepper() {
  // Access the angle value directly
  float angle = message;

  // Calculate the target position in steps using the conversion factor
  int targetPosition = angle * stepsPerRevolution/360;

  // Set the motor's target position using the calculated value
  stepper.moveTo(targetPosition);

  // Run the motor until it reaches the target position
  while (abs(stepper.distanceToGo()) > 0) stepper.run();
}

void move_stepper_relative() {
  float relative_angle = message;
  
  // Get the current position (implementation depends on your stepper library)
  int current_position = stepper.currentPosition();

  // Calculate the target position based on relative angle and current position
  int target_position = current_position + (relative_angle * stepsPerRevolution/360);

  // Move the motor to the calculated target position
  stepper.moveTo(target_position);
  while (abs(stepper.distanceToGo()) > 0) stepper.run();
}
