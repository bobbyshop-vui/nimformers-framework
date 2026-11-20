import nimformers

let group = initProcessGroup(["cpu"])
echo group.partition(12)
