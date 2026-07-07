classdef PololuMaestro < handle
    properties
        Port
    end

    properties (Access = private)
        Velocity = nan(1, 12)
        Acceleration = nan(1, 12)
    end

    methods
        function obj = PololuMaestro(portString)
            obj.Port = ArCOMObject_Bpod(portString, 9600);
            for channel = 0:5
                obj.setSpeed(channel, 128);
                obj.setAcceleration(channel, 121);
            end
        end

        function setMotor(obj, motorID, position, velocity, acceleration)
            if nargin < 4
                velocity = 1;
            end
            if nargin < 5
                acceleration = 1;
            end
            velocity = min(1, max(0, velocity));
            acceleration = min(1, max(0, acceleration));

            speed = round(2 ^ (velocity * 7));
            accel = round(2 ^ (acceleration * 7));
            index = motorID + 1;
            if obj.Velocity(index) ~= velocity
                obj.setSpeed(motorID, speed);
                obj.Velocity(index) = velocity;
            end
            if obj.Acceleration(index) ~= acceleration
                obj.setAcceleration(motorID, accel);
                obj.Acceleration(index) = acceleration;
            end

            target = round((1000 + 1000 * ((position + 1) / 2)) * 4);
            obj.writeCommand(132, motorID, target);
        end

        function delete(obj)
            obj.Port = [];
        end
    end

    methods (Access = private)
        function setSpeed(obj, channel, value)
            obj.writeCommand(135, channel, value);
        end

        function setAcceleration(obj, channel, value)
            obj.writeCommand(137, channel, value);
        end

        function writeCommand(obj, command, channel, value)
            lowBits = bitand(uint16(value), 127);
            highBits = bitand(bitshift(uint16(value), -7), 127);
            obj.Port.write(uint8([command channel lowBits highBits]), 'uint8');
        end
    end
end
