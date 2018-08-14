#!/usr/bin/env bash
# Options:
#   -i: Required, name of the image that you want to build, hosted on reg.ihme.uw.edu. For example, core/rstudio:3.4.3.1
#   -f: Required, path to the file where you want to save your image
#   -t: number of threads to use when pulling Dockerfile and building Singularity image. Default: 2. 
#       Only applies on cluster-prod -- on cluster-dev, will use all CPUs assigned to your qlogin session.

## Example:
# sh /home/j/temp/grant/build_singularity.sh -i core/rstudio:3.4.3.1 -f /share/singularity-images/test_deploy/shell_test2.img -t 8

# Set arguments for options
while getopts "i::f::t::" opt; do
  case $opt in
    i ) image_name=$OPTARG
    ;;
    f ) file_path=$OPTARG
    ;;
    t ) threads=$OPTARG
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


# function to set environmental variables for build-time configuration
# Mostly to constrain the number of python threads and to ensure that Singularity does not save cached images to a user's H drive.
function set_env_variables {
  echo "Setting environmental variables"
  export SINGULARITY_CACHEDIR="/tmp"
  export SINGULARITY_CACHE="/tmp"
  export SINGULARITY_DISABLE_CACHE="true"
  export SINGULARITY_PYTHREADS="$threads"
}

# function to submit a new job
function build_singularity {
  if [ -z $threads ]
    then
      threads=2
  fi 

  if [ -z $image_name ]
    then
      echo "Error: Image name must be specified using -i flag, e.g. -i core/rstudio:3.4.3.1"
      exit 1
  fi 

  if [ -z $file_path ]
    then
      echo "Error: Destination file path must be specified using -f flag, e.g. -f /share/singularity-images/my_image_dir/my_image.img"
      exit 1
    else
      target_directory=$(dirname "${file_path}")
      echo $target_directory
      if [ ! -d "$target_directory" ] # If destination folder doesn't exist, then error
        then
          echo "Error: Folder for destination file path must first exist: $target_directory"
          exit 1
      fi
  fi 

  build_host=$HOSTNAME

  if [ "${build_host}" == "cluster-prod.ihme.washington.edu" ] || [ "${build_host}" == "cluster-dev.ihme.washington.edu" ]
    then
      echo "Error: Cannot build Singularity on the submit host -- use qlogin to get an interactive login session"
	  exit 1
    fi

  set_env_variables

  # Pull down current umask, convert to Singularity-friendly umask (see https://github.com/singularityware/singularity/issues/1079)
  # The rest is within large brackets to ensure that umask gets reset if any of the commands below fail
  current_umask=$(umask)
  umask 022

  {
    ## Set CPUs to pin to using numactl -- for dev cluster, CPU enforcement is enabled so we can pull the CPUs directly. For cluster-prod, need to generate a list
    if [ $SGE_CLUSTER_NAME == "devnotgonnahappen" ]
      then
        cpu_list="$(cat /proc/$$/status | grep '^Cpus_allowed_list:' | awk '{print $NF}')"
      else
		target_thread=$(($threads - 1))
        cpu_list="0-$target_thread"
      fi

    echo "numactl -C +$cpu_list /share/local/singularity-2.4.2/bin/singularity build $file_path docker://reg.ihme.uw.edu/$image_name"
	numactl -C +$cpu_list /share/local/singularity-2.4.2/bin/singularity build $file_path docker://reg.ihme.uw.edu/$image_name
    # echo "chmod 744 $file_path"
	chmod 744 $file_path
    umask $current_umask
  } || {
    umask $current_umask
  }

}

# call the submission function
build_singularity
