clear; close all; clc;
matRad_rc;

planFile = 'plan_reference.mat';
patientFiles = {'patient_real_ct.mat', 'patient_baseline.mat', 'patient_dir_cbct.mat', 'patient_dir_ct.mat'};

plan = load(planFile, 'stf', 'pln', 'w');

fprintf('=== CONSISTENCY CHECK ===\n');
doseFiles = cell(size(patientFiles));
for i = 1:numel(patientFiles)
    [~, name] = fileparts(patientFiles{i});
    doseFiles{i} = [name '_dose.mat'];

    if ~isfile(doseFiles{i})
        error('Missing dose file: %s', doseFiles{i});
    end

    check = load(doseFiles{i}, 'w');
    if ~isequal(check.w, plan.w)
        error('Weights in %s differ from the reference plan. Export aborted.', doseFiles{i});
    end
    fprintf('  [OK] %s\n', doseFiles{i});
end
clear check;

fprintf('\n=== DICOM EXPORT ===\n');
for i = 1:numel(patientFiles)
    [~, name] = fileparts(patientFiles{i});
    fprintf('[%d/%d] %s\n', i, numel(patientFiles), name);

    data = load(patientFiles{i}, 'ct', 'cst');
    dose = load(doseFiles{i}, 'physicalDose');

    exportDir = fullfile(pwd, [name '_dicom']);
    if ~isfolder(exportDir)
        mkdir(exportDir);
    end

    dcmExport = matRad_DicomExporter();
    dcmExport.dicomDir = exportDir;
    dcmExport.ct = data.ct;
    dcmExport.cst = data.cst;
    dcmExport.stf = plan.stf;
    dcmExport.pln = plan.pln;
    dcmExport.resultGUI = struct('physicalDose', dose.physicalDose, 'w', plan.w);
    dcmExport.enableRtPlanExport = false;
    dcmExport.matRad_exportDicom();

    clear data dose dcmExport;
end

fprintf('=== EXPORT COMPLETED ===\n');
