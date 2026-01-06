# Simple Agent AI Example

def vacuum_agent(location, is_dirty):
    if is_dirty:
        return f"Location {location}: CLEAN"
    else:
        return f"Location {location}: MOVE"

# Environment states
rooms = {
    "A": True,
    "B": False
}

for room, dirt in rooms.items():
    print(vacuum_agent(room, dirt))
