function LearnFacadeObjects
%% user settings
% objNames = {'window', 'door'};
% thres = [2.0 -0.3];
objNames = {'window'};
thres = [-0.3];

testImgFN = '/data/v50/sangdonp/FacadeRecognition/data/Paris/pascalFormat_1/VOC2010/VOCdevkit/Paris/JPEGImages/monge_40.jpg';

%% learn and evaluate
for oInd=1:numel(objNames)
    %learn
    objName = objNames{oInd};
    
    pascal(objName, 1);
    print(gcf, '-dpdf', sprintf('pr_curve_%s.pdf', objName));
    keyboard;
    
%     load(sprintf('/data/v50/sangdonp/FacadeRecognition/data/Paris/pascalFormat_2/learnObjectUsingDPM5/2010/%s_final.mat', objName));
% 
%     model.vis = @() visualizemodel(model, ...
%                       1:2:length(model.rules{model.start}));
%     test(testImgFN, model, thres(oInd));
end
end

function test(imname, model, thresh)
cls = model.class;
fprintf('///// Running demo for %s /////\n\n', cls);

% load and display image
im = imread(imname);
clf;
image(im);
axis equal; 
axis on;
disp('input image');
disp('press any key to continue'); pause;
disp('continuing...');

% load and display model
model.vis();
disp([cls ' model visualization']);
print(gcf, '-dpdf', sprintf('vis_model_%s.pdf', cls));
disp('press any key to continue'); pause;
disp('continuing...');

% detect objects
[ds, bs] = imgdetect(im, model, thresh);
top = nms(ds, 0.5);
clf;
if model.type == model_types.Grammar
  bs = [ds(:,1:4) bs];
end
showboxes(im, reduceboxes(model, bs(top,:)));
disp('detections');
print(gcf, '-dpdf', sprintf('detections_%s.pdf', cls));
disp('press any key to continue'); pause;
disp('continuing...');

% if model.type == model_types.MixStar
%   % get bounding boxes
%   bbox = bboxpred_get(model.bboxpred, ds, reduceboxes(model, bs));
%   bbox = clipboxes(im, bbox);
%   top = nms(bbox, 0.5);
%   clf;
%   showboxes(im, bbox(top,:));
%   print(gcf, '-dpdf', sprintf('bbs_%s.pdf', cls));
%   disp('bounding boxes');
%   disp('press any key to continue'); pause;
% end

fprintf('\n');

end