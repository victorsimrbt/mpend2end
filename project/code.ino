#include <Stepper.h>

const int stepsPerRevolution = 200; // change this to match your stepper motor

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, 8, 9, 10, 11);

const float radiansPerStep = 2 * PI / stepsPerRevolution;
const float rotationAngle = 2 * PI;
const float stepRadians = 0.1;

void setup() {
  myStepper.setSpeed(60);
  Serial.begin(9600);
}

void loop() {
  float currentAngle = 0;


  while (currentAngle < 2 * rotationAngle) {
    Serial.print("Current Angle: ");
    Serial.println(currentAngle);
    
    int steps = stepRadians / radiansPerStep;
    myStepper.step(steps);
    currentAngle += stepRadians;
    
    delay(1000);
  }

  int totalSteps = -2 * stepsPerRevolution; 
  myStepper.setSpeed(200); 
  myStepper.step(totalSteps);

  Serial.println("Completed rotations.");
  while(true);
}
