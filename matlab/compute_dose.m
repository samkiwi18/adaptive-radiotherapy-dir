clear; close all; clc;
matRad_rc;

refFile = 'patient_real_ct.mat';
warpedFiles = {'patient_baseline.mat', 'patient_dir_cbct.mat', 'patient_dir_ct.mat'};
planFile = 'plan_reference.mat';
isoTargetName = 'PTV';

doseGridRes = 3;
runSequencing = true;
sequencingLevel = 7;
forceReoptimization = false;
forceRecalculation = false;

if isfile(planFile) && ~forceReoptimization
    fprintf('Loading existing plan: %s\n', planFile);
    plan = load(planFile, 'stf', 'pln', 'w');
else
    fprintf('=== PLAN OPTIMIZATION ON %s ===\n', refFile);
    ref = load(refFile, 'ct', 'cst', 'pln');
    ref.pln.propDoseCalc.doseGrid.resolution = struct('x', doseGridRes, 'y', doseGridRes, 'z', doseGridRes);
    ref.pln.propSeq.runSequencing = runSequencing;
    ref.pln.propSeq.sequencingLevel = sequencingLevel;

    isoIdx = find(strcmpi(ref.cst(:, 2), isoTargetName), 1);
    if isempty(isoIdx)
        error('Structure "%s" not found in cst.', isoTargetName);
    end
    cstIso = ref.cst;
    cstIso(:, 3) = {'OAR'};
    cstIso{isoIdx, 3} = 'TARGET';
    ref.pln.propStf.isoCenter = repmat(matRad_getIsoCenter(cstIso, ref.ct, false), numel(ref.pln.propStf.gantryAngles), 1);
    clear cstIso;
    fprintf('Isocenter (%s centroid): [%.1f %.1f %.1f] mm\n', isoTargetName, ref.pln.propStf.isoCenter(1, :));

    ref.stf = matRad_generateStf(ref.ct, ref.cst, ref.pln);

    fprintf('Computing dose influence matrix...\n');
    dij = matRad_calcDoseInfluence(ref.ct, ref.cst, ref.stf, ref.pln);

    fprintf('Optimizing fluence...\n');
    resultGUI = matRad_fluenceOptimization(dij, ref.cst, ref.pln);

    if runSequencing
        fprintf('Leaf sequencing...\n');
        wFluence = resultGUI.w;
        resultGUI = matRad_sequencing(resultGUI, ref.stf, dij, ref.pln);
    end
    clear dij;

    plan.stf = ref.stf;
    plan.pln = ref.pln;
    plan.w = resultGUI.w;
    if isfield(resultGUI, 'wSequenced')
        plan.w = resultGUI.wSequenced;
    end
    if runSequencing
        fprintf('Relative change of weights after sequencing: %.3f\n', norm(plan.w - wFluence) / norm(wFluence));
        clear wFluence;
    end
    clear ref resultGUI;

    save(planFile, '-struct', 'plan', '-v7');
    fprintf('Plan saved: %s\n', planFile);
end

fprintf('\n=== FORWARD DOSE CALCULATION (fixed weights) ===\n');

allFiles = [{refFile}, warpedFiles];

for i = 1:numel(allFiles)
    [~, name] = fileparts(allFiles{i});
    outFile = [name '_dose.mat'];

    if isfile(outFile) && ~forceRecalculation
        fprintf('[%d/%d] %s already computed, skipped\n', i, numel(allFiles), name);
        continue;
    end

    fprintf('[%d/%d] %s\n', i, numel(allFiles), name);
    data = load(allFiles{i}, 'ct', 'cst');

    res = matRad_calcDoseForward(data.ct, data.cst, plan.stf, plan.pln, plan.w);

    out.physicalDose = res.physicalDose;
    out.w = plan.w;
    out.sourceFile = allFiles{i};
    save(outFile, '-struct', 'out', '-v7');

    clear data res out;
end

fprintf('=== DONE ===\n');
