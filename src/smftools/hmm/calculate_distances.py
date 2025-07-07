# calculate_distances

def calculate_distances(intervals, threshold=0.9):
    """
    Calculates distance between features in a read.
    Takes in a list of intervals (start of feature, length of feature)
    """
    # Sort intervals by start position
    intervals = sorted(intervals, key=lambda x: x[0])
    intervals = [interval for interval in intervals if interval[2] > threshold]
    
    # Calculate distances
    distances = []
    for i in range(len(intervals) - 1):
        end_current = intervals[i][0] + intervals[i][1]
        start_next = intervals[i + 1][0]
        distances.append(start_next - end_current)
    return distances