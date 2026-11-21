clear; close all; clc;
matRad_rc;

planFile = 'plan_reference.mat';
patientFile = 'patient_real_ct.mat';
targetName = 'PTV';
huWindow = [-1000 400];

plan = load(planFile, 'pln', 'stf');
data = load(patientFile, 'ct', 'cst');
ct = data.ct;
cst = data.cst;

iso = plan.pln.propStf.isoCenter(1, :);
stfIso = vertcat(plan.stf.isoCenter);
fprintf('pln isocenter: [%.2f %.2f %.2f] mm\n', iso);
fprintf('Max isocenter deviation among %d beams: %.3f mm\n', numel(plan.stf), max(vecnorm(stfIso - iso, 2, 2)));

if isfield(ct, 'x') && ~isempty(ct.x)
    xVec = ct.x(:)';
    yVec = ct.y(:)';
    zVec = ct.z(:)';
else
    xVec = (1:ct.cubeDim(2)) * ct.resolution.x;
    yVec = (1:ct.cubeDim(1)) * ct.resolution.y;
    zVec = (1:ct.cubeDim(3)) * ct.resolution.z;
end

[~, ix] = min(abs(xVec - iso(1)));
[~, iy] = min(abs(yVec - iso(2)));
[~, iz] = min(abs(zVec - iso(3)));
fprintf('Isocenter voxel (row, col, slice): [%d %d %d]\n', iy, ix, iz);

targetIdx = find(strcmpi(cst(:, 2), targetName), 1);
mask = false(ct.cubeDim);
if ~isempty(targetIdx)
    mask(cst{targetIdx, 4}{1}) = true;
    [r, c, s] = ind2sub(ct.cubeDim, find(mask));
    centroid = [mean(xVec(c)) mean(yVec(r)) mean(zVec(s))];
    fprintf('%s centroid: [%.2f %.2f %.2f] mm\n', targetName, centroid);
    fprintf('Distance isocenter - %s centroid: %.2f mm\n', targetName, norm(iso - centroid));
    if mask(iy, ix, iz)
        fprintf('Isocenter lies inside %s\n', targetName);
    else
        fprintf('WARNING: isocenter lies outside %s\n', targetName);
    end
else
    warning('Structure "%s" not found in cst.', targetName);
end

hu = ct.cubeHU{1};

views = {
    'Axial',    hu(:, :, iz),            mask(:, :, iz),            xVec, yVec, iso(1), iso(2), 'x [mm]', 'y [mm]', 'reverse';
    'Coronal',  squeeze(hu(iy, :, :))',  squeeze(mask(iy, :, :))',  xVec, zVec, iso(1), iso(3), 'x [mm]', 'z [mm]', 'normal';
    'Sagittal', squeeze(hu(:, ix, :))',  squeeze(mask(:, ix, :))',  yVec, zVec, iso(2), iso(3), 'y [mm]', 'z [mm]', 'normal'
    };

fig = figure('Name', 'Plan isocenter', 'Color', 'w', 'Position', [100 100 1500 520]);
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for v = 1:size(views, 1)
    nexttile;
    imagesc(views{v, 4}, views{v, 5}, views{v, 2}, huWindow);
    colormap(gca, gray);
    axis image;
    set(gca, 'YDir', views{v, 10});
    hold on;
    if any(views{v, 3}(:))
        contour(views{v, 4}, views{v, 5}, double(views{v, 3}), [0.5 0.5], 'r', 'LineWidth', 1.5);
    end
    xline(views{v, 6}, 'y--', 'LineWidth', 1);
    yline(views{v, 7}, 'y--', 'LineWidth', 1);
    plot(views{v, 6}, views{v, 7}, 'c+', 'MarkerSize', 18, 'LineWidth', 2.5);
    title(views{v, 1});
    xlabel(views{v, 8});
    ylabel(views{v, 9});
    hold off;
end

sgtitle(sprintf('Isocenter [%.1f %.1f %.1f] mm  |  red: %s', iso, targetName));
exportgraphics(fig, 'isocenter_check.png', 'Resolution', 200);
