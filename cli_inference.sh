#!/bin/bash

# Mapperatorinator CLI - Interactive Inference Script
# Based on web-ui.py functionality

set -e  # Exit on error

# Colors for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored text
print_color() {
    local color=$1
    local text=$2
    echo -e "${color}${text}${NC}"
}

# Function to print section headers
print_header() {
    echo
    print_color $CYAN "======================================"
    print_color $CYAN "$1"
    print_color $CYAN "======================================"
    echo
}

# Function to prompt for input with default value
prompt_input() {
    local prompt=$1
    local default=$2
    local var_name=$3
    
    if [ -n "$default" ]; then
        read -p "$(print_color $GREEN "$prompt") [default: $default]: " input
        if [ -z "$input" ]; then
            input="$default"
        fi
    else
        read -p "$(print_color $GREEN "$prompt"): " input
    fi
    
    eval "$var_name='$input'"
}

# Function to prompt for yes/no
prompt_yn() {
    local prompt=$1
    local default=$2
    local var_name=$3
    
    while true; do
        if [ "$default" = "y" ]; then
            read -p "$(print_color $GREEN "$prompt") [Y/n]: " yn
            yn=${yn:-y}
        else
            read -p "$(print_color $GREEN "$prompt") [y/N]: " yn
            yn=${yn:-n}
        fi
        
        case $yn in
            [Yy]* ) eval "$var_name=true"; break;;
            [Nn]* ) eval "$var_name=false"; break;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to prompt for multiple choice
prompt_choice() {
    local prompt=$1
    local var_name=$2
    shift 2
    local options=("$@")
    
    while true; do
        print_color $GREEN "$prompt"
        for i in "${!options[@]}"; do
            echo "  $((i+1))) ${options[i]}"
        done
        read -p "Select option (1-${#options[@]}): " choice
        
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
            eval "$var_name='${options[$((choice-1))]}'"
            break
        else
            print_color $RED "Invalid choice. Please select 1-${#options[@]}."
        fi
    done
}

# Function to prompt for multiple selection using arrow keys and spacebar
prompt_multiselect() {
    local prompt=$1
    local var_name=$2
    shift 2
    local options=("$@")
    local num_options=${#options[@]}
    local selections=()
    for (( i=0; i<num_options; i++ )); do
        selections[i]=0
    done
    local current_idx=0

    # Hide cursor for a cleaner UI
    tput civis
    # Ensure cursor is shown again on exit
    trap 'tput cnorm; return' EXIT

    # Initial draw
    tput clear
    
    while true; do
        # Move cursor to top left
        tput cup 0 0

        echo -e "${GREEN}${prompt}${NC}"
        echo "(Use UP/DOWN to navigate, SPACE to select/deselect, ENTER to confirm)"

        for i in "${!options[@]}"; do
            local checkbox="[ ]"
            if [[ ${selections[i]} -eq 1 ]]; then
                checkbox="[${GREEN}x${NC}]"
            fi

            if [ "$i" -eq "$current_idx" ]; then
                echo -e "  ${CYAN}> $checkbox ${options[i]}${NC}"
            else
                echo -e "    $checkbox ${options[i]}"
            fi
        done
        # Clear rest of the screen
        tput ed

        # Read a single keystroke.
        # IFS= ensures space is read as a character, not a delimiter.
        IFS= read -rsn1 key
        
        # Handle escape sequences for arrow keys
        if [[ "$key" == $'\e' ]]; then
            read -rsn2 -t 0.1 key
        fi

        case "$key" in
            '[A') # Up arrow
                current_idx=$(( (current_idx - 1 + num_options) % num_options ))
                ;;
            '[B') # Down arrow
                current_idx=$(( (current_idx + 1) % num_options ))
                ;;
            ' ') # Space bar
                if [[ ${selections[current_idx]} -eq 1 ]]; then
                    selections[current_idx]=0
                else
                    selections[current_idx]=1
                fi
                ;;
            '') # Enter key
                break
                ;;
        esac
    done

    # Show cursor again and clear the trap
    tput cnorm
    trap - EXIT

    # Go back to the bottom of the screen
    tput cup $(tput lines) 0
    clear # Clean up the interactive menu from screen

    # Collect selected options
    local selected_options=()
    for i in "${!options[@]}"; do
        if [[ ${selections[i]} -eq 1 ]]; then
            selected_options+=("${options[i]}")
        fi
    done

    # Format the result list for Hydra/Python: '["item1", "item2"]'
    if [ ${#selected_options[@]} -gt 0 ]; then
        local formatted_items=""
        for item in "${selected_options[@]}"; do
            if [ -n "$formatted_items" ]; then
                # Each item is wrapped in double quotes
                formatted_items="$formatted_items,\"$item\""
            else
                formatted_items="\"$item\""
            fi
        done
        # The whole list is wrapped in brackets
        eval "$var_name='[$formatted_items]'"
    else
        # Return an empty string if nothing is selected
        eval "$var_name=''"
    fi
}


# Function to validate file path
validate_file() {
    local file_path=$1
    if [ ! -f "$file_path" ]; then
        print_color $RED "File not found: $file_path"
        return 1
    fi
    return 0
}

# Main script starts here
print_color $PURPLE "╔═══════════════════════════════════════════╗"
print_color $PURPLE "║            Mapperatorinator CLI           ║"
print_color $PURPLE "║        Interactive Inference Setup        ║"
print_color $PURPLE "╚═══════════════════════════════════════════╝"
echo

# 2. Required Paths
print_header "Required Paths"

# Python Path
prompt_input "Python executable path" "python" python_executable

# Audio Path (Required)
while true; do
    prompt_input "Audio file path (required)" "input/demo.mp3" audio_path
    if [ -z "$audio_path" ]; then
        print_color $RED "Audio path is required!"
        continue
    fi
    if validate_file "$audio_path"; then
        break
    fi
done

# Output Path
prompt_input "Output directory path" "$(dirname "$audio_path")" output_path

# Beatmap Path (Optional)
prompt_input "Beatmap file path (optional, for in-context learning)" "" beatmap_path
if [ -n "$beatmap_path" ] && ! validate_file "$beatmap_path"; then
    print_color $YELLOW "Warning: Beatmap file not found, continuing without it"
    beatmap_path=""
fi

# 3. Basic Settings
print_header "Basic Settings"

# Game Mode (MODIFIED BLOCK)
gamemode_options=("osu!" "Taiko" "Catch" "Mania")
while true; do
    print_color $GREEN "Game mode:"
    for i in "${!gamemode_options[@]}"; do
        echo "  $i) ${gamemode_options[$i]}"
    done
    read -p "$(print_color $GREEN "Select option (0-3)") [default: 0]: " gamemode_input
    # Set default value to 0 if input is empty
    gamemode=${gamemode_input:-0}

    if [[ "$gamemode" =~ ^[0-3]$ ]]; then
        break
    else
        print_color $RED "Invalid choice. Please select a number between 0 and 3."
        echo # Add a blank line for spacing before re-prompting
    fi
done

# Difficulty
prompt_input "Difficulty (1.0-10.0)" "5.5" difficulty

# Year
# default is 2023, and 2007-2023 are valid years
prompt_input "Year" "2023" year
if ! [[ "$year" =~ ^(200[7-9]|201[0-9]|202[0-3])$ ]]; then
    print_color $RED "Invalid year! Year must be between 2007 and 2023. Defaulting to 2023."
    year=2023
fi

# 4. Advanced Settings (Optional)
print_header "Advanced Settings (Optional - Press Enter to skip)"
print_color $BLUE "Difficulty Settings:"
prompt_input "HP Drain Rate (0-10)" "" hp_drain_rate
prompt_input "Circle Size (0-10)" "" circle_size
prompt_input "Overall Difficulty (0-10)" "" overall_difficulty
prompt_input "Approach Rate (0-10)" "" approach_rate
print_color $BLUE "Slider Settings:"
prompt_input "Slider Multiplier" "" slider_multiplier
prompt_input "Slider Tick Rate" "" slider_tick_rate
if [ "$gamemode" -eq 3 ]; then
    print_color $BLUE "Mania Settings:"
    prompt_input "Key Count" "" keycount
    prompt_input "Hold Note Ratio (0-1)" "" hold_note_ratio
    prompt_input "Scroll Speed Ratio" "" scroll_speed_ratio
fi
print_color $BLUE "Generation Settings:"
prompt_input "CFG Scale (1-20)" "" cfg_scale
prompt_input "Temperature (0-2)" "" temperature
prompt_input "Top P (0-1)" "" top_p
prompt_input "Seed (random if empty)" "" seed
prompt_input "Mapper ID" "" mapper_id
print_color $BLUE "Timing Settings:"
prompt_input "Start Time (seconds)" "" start_time
prompt_input "End Time (seconds)" "" end_time

# 5. Boolean Options
print_header "Export & Processing Options"
prompt_yn "Export as .osz file?" "n" export_osz
prompt_yn "Add to existing beatmap?" "n" add_to_beatmap
prompt_yn "Add hitsounds?" "n" hitsounded
prompt_yn "Use super timing analysis?" "n" super_timing

# 6. Descriptors
print_header "Style Descriptors"

# Positive descriptors with interactive multi-select
descriptor_options=("jump aim" "stream" "tech" "aim" "speed" "flow" "clean" "complex" "simple" "modern" "classic" "spaced" "stacked")
prompt_multiselect "Positive descriptors (describe desired mapping style):" descriptors "${descriptor_options[@]}"

# Negative descriptors with interactive multi-select
prompt_multiselect "Negative descriptors (styles to avoid):" negative_descriptors "${descriptor_options[@]}"

# In-context options (only if beatmap is provided)
if [ -n "$beatmap_path" ]; then
    print_header "In-Context Learning Options"
    context_options_list=("timing" "patterns" "structure" "style")
    prompt_multiselect "In-context learning aspects:" in_context_options "${context_options_list[@]}"
fi


# 7. Build and Execute Command
print_header "Command Generation"

# Start building the command
cmd_args=("$python_executable" "inference.py") 

# Helper function to add argument. Wraps value in single quotes.
add_arg() {
    local key=$1
    local value=$2
    if [ -n "$value" ]; then
        # This format 'key=value' is robust for Hydra, even with complex values
        # like lists represented as strings: descriptors='["item1", "item2"]'
        cmd_args+=("${key}=${value}") # Removed extra quotes for direct execution
    fi
}

# Helper function to add boolean argument
add_bool_arg() {
    local key=$1
    local value=$2
    if [ "$value" = "true" ]; then
        cmd_args+=("${key}=true")
    else
        cmd_args+=("${key}=false")
    fi
}

# Add all arguments
add_arg "audio_path" "'$audio_path'"
add_arg "output_path" "'$output_path'"
add_arg "beatmap_path" "'$beatmap_path'"
add_arg "gamemode" "$gamemode"
add_arg "difficulty" "$difficulty"
add_arg "year" "$year"

# Optional numeric parameters
add_arg "hp_drain_rate" "$hp_drain_rate"
add_arg "circle_size" "$circle_size"
add_arg "overall_difficulty" "$overall_difficulty"
add_arg "approach_rate" "$approach_rate"
add_arg "slider_multiplier" "$slider_multiplier"
add_arg "slider_tick_rate" "$slider_tick_rate"
add_arg "keycount" "$keycount"
add_arg "hold_note_ratio" "$hold_note_ratio"
add_arg "scroll_speed_ratio" "$scroll_speed_ratio"
add_arg "cfg_scale" "$cfg_scale"
add_arg "temperature" "$temperature"
add_arg "top_p" "$top_p"
add_arg "seed" "$seed"
add_arg "mapper_id" "$mapper_id"
add_arg "start_time" "$start_time"
add_arg "end_time" "$end_time"

# List parameters (now correctly quoted)
add_arg "descriptors" "$descriptors"
add_arg "negative_descriptors" "$negative_descriptors"
add_arg "in_context" "$in_context_options"

# Boolean parameters
add_bool_arg "export_osz" "$export_osz"
add_bool_arg "add_to_beatmap" "$add_to_beatmap"
add_bool_arg "hitsounded" "$hitsounded"
add_bool_arg "super_timing" "$super_timing"


# Display the command
print_color $YELLOW "Generated command:"
echo
# Use printf for safer printing of arguments
printf "%s " "${cmd_args[@]}"
echo
echo

# Ask for confirmation
prompt_yn "Execute this command?" "y" execute_cmd

if [ "$execute_cmd" = "true" ]; then
    print_header "Executing Inference"
    print_color $GREEN "Starting inference process..."
    echo
    
    # Execute the command by expanding the array. No need for eval.
    "${cmd_args[@]}"
    
    exit_code=$?
    echo
    if [ $exit_code -eq 0 ]; then
        print_color $GREEN "✓ Inference completed successfully!"
    else
        print_color $RED "✗ Inference failed with exit code: $exit_code"
    fi
else
    print_color $YELLOW "Command generation cancelled."
    echo
    print_color $BLUE "You can copy and run the command manually:"
    # Use printf for safer printing of arguments
    printf "%s " "${cmd_args[@]}"
    echo
fi

echo
print_color $PURPLE "Thank you for using Mapperatorinator CLI!"