# frozen_string_literal: true

# Requires cumo (https://github.com/sonots/cumo), falling back to
# Numo::NArray -- cumo's CPU-only, API-compatible counterpart -- when no CUDA
# GPU is available, so the scripts in this directory can still be exercised
# end to end without GPU hardware. See this directory's README. Shared by
# simplify_and_run.rb and train_and_export.rb.
begin
  require 'cumo/narray'
rescue LoadError, RuntimeError => e
  # LoadError: the gem isn't installed. RuntimeError (or a subclass): the gem
  # is installed but its native extension couldn't find a CUDA-capable GPU at
  # require time (e.g. "CUDA driver version is insufficient").
  warn "cumo is not available (#{e.message}); install it on a CUDA-capable " \
       "machine -- see this directory's README. Falling back to Numo::NArray " \
       "(cumo's CPU-only, API-compatible counterpart) so this sample can " \
       'still be exercised.'
  require 'numo/narray'
  Cumo = Numo unless defined?(Cumo)
end
