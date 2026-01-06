# Script from https://github.com/snuvclab/gala/blob/main/scripts/setup.sh


# SMPL-X
# Script from https://github.com/yfeng95/SCARF/blob/main/fetch_data.sh
mkdir -p ./deformer/data
# SMPL-X 2020 (neutral SMPL-X model with the FLAME 2020 expression blendshapes)
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)
# wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O './deformer/data/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# Download smpl-x model
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1' -O './data/models_smplx_v1_1.zip' --no-check-certificate --continue
unzip data/models_smplx_v1_1.zip -d data
rm data/models_smplx_v1_1.zip

# scarf utilities
echo -e "\nDownloading data..."
wget https://owncloud.tuebingen.mpg.de/index.php/s/n58Fzbzz7Ei9x2W/download -O ./deformer/data/scarf_utilities.zip
unzip ./deformer/data/scarf_utilities.zip -d ./deformer/data
rm ./deformer/data/scarf_utilities.zip