function uid = dicomuid()
% Octave replacement for dicomuid: random UID under the DICOM root, not globally unique
uid = sprintf('1.2.840.10008.5.1.4.1.1.%d.%d', randi([10000, 99999]), randi([10000, 99999]));
end