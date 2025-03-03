import os
import shutil
from pathlib import Path

def copy_yaml_files(input_dir, target_dir):
    # Create target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Walk through the input directory
    for root, dirs, files in os.walk(input_dir):
        # Filter files with desired extensions
        yaml_file = None
        xml_file = None
        inp_file = None
        therm_file = None
        cti_file = None

        for file in files:
            if file.endswith('.yaml'):
                yaml_file = file
            elif file.endswith('.xml'):
                xml_file = file
            elif file.endswith('.inp'):
                inp_file = file
            elif file.endswith('.dat') and 'therm' in file:
                therm_file = file
            elif file.endswith('.cti'):
                cti_file = file
        # print(f"root: {root}", yaml_file, xml_file, inp_file, therm_file, cti_file)
        ##create direcotry for each chemistry
        if yaml_file is not None or \
            xml_file is not None or \
            inp_file is not None:

            rel_path = os.path.relpath(root, input_dir)
            # Create target subdirectory if needed
            target_root = os.path.join(target_dir, rel_path)
            if not os.path.exists(target_root):
                os.makedirs(target_root)
            if os.path.exists(os.path.join(target_root, "chem.yaml")):
                continue
            if yaml_file is not None:
                source_file = os.path.join(root, yaml_file)
                target_file = os.path.join(target_root, "chem.yaml")
                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")
                continue

            exit_code = 0

            if xml_file is not None:
                source_file = os.path.join(root, xml_file)
                target_file = os.path.join(target_root, "chem.yaml")
                exit_code = os.system(f"ctml2yaml '{source_file}' '{target_file}'  > /dev/null 2>&1")
                if exit_code == 0:
                    print(f"Converted: {source_file} -> {target_file}")
                    continue
                

            if cti_file is not None:
                source_file = os.path.join(root, cti_file)
                target_file = os.path.join(target_root, "chem.yaml")
                exit_code = os.system(f"cti2yaml '{source_file}' '{target_file}'  > /dev/null 2>&1")
                if exit_code == 0:
                    print(f"Converted: {source_file} -> {target_file}")
                    continue
            
            if inp_file is not None and therm_file is not None:
                src_inp_file = os.path.join(root, inp_file)
                src_therm_file = os.path.join(root, therm_file)
                target_file = os.path.join(target_root, "chem.yaml")
                exit_code = os.system(f"ck2yaml --input '{src_inp_file}' --thermo '{src_therm_file}' --output '{target_file}' --permissive > /dev/null 2>&1")
                if exit_code == 0:
                    print(f"Converted: {src_inp_file} -> {target_file}")
                    continue
            if exit_code != 0:
                print(f"Failed to convert: {root}")

if __name__ == "__main__":
    # Get input and target directories from user
    input_directory = "./ext_repo"
    target_directory = "./ck_files"
    
    # Convert to absolute paths
    input_directory = os.path.abspath(input_directory)
    target_directory = os.path.abspath(target_directory)
    
    # Copy the files
    copy_yaml_files(input_directory, target_directory)
    print("Done!")