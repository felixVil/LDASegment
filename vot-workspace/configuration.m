set_global_variable('experiment_cache', false);
set_global_variable('workspace_path', fileparts(mfilename('fullpath')));

set_global_variable('version', 6);
set_global_variable('trax_timeout', 5000);
set_global_variable('licFilePath', '');
% Enable more verbose output
% set_global_variable('debug', 1);

% Disable result caching
% set_global_variable('cache', 0);

% Select experiment stack
set_global_variable('stack', 'vot2018');
