# Model-Based Agent Example (Vacuum Cleaner)

# Internal model (memory)
room_status = {
    "A": "dirty",
    "B": "unknown"
}

def model_based_agent(location, perception):
    # Update internal model
    room_status[location] = perception

    # Decide action
    if room_status[location] == "dirty":
        return "CLEAN"
    else:
        return "MOVE"

# Environment simulation
print("Room A:", model_based_agent("A", "dirty"))
print("Room A:", model_based_agent("A", "clean"))
print("Room B:", model_based_agent("B", "dirty"))
