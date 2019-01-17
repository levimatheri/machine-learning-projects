import math
import sys

class State:
    def __init__(self, layout, action, parent, depth):
        self.layout = layout
        self.action = action
        self.parent = parent
        self.depth = depth
        self.blank_pos = layout.index(0)
        self.n = int(math.sqrt(len(layout)))
        self.children = []

    def __hash__(self):
        multi = 1
        result = 0
        for i in reversed(self.layout):
            result += (i * multi)
            multi *= 10
        return result

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and
                self.layout == other.layout)

    def valid_moves(self):
        blank_row = int(self.blank_pos / self.n)
        blank_col = self.blank_pos % self.n
        moves = []

        # check if we can go up. Check if row of blank is 0
        if blank_row != 0:
            moves.append("Up")
        # check if we can go down. Check if row of blank is n-1
        if blank_row != self.n - 1:
            moves.append("Down")
        # check if we can go left. Check if col of blank is 0
        if blank_col != 0:
            moves.append("Left")
        # check if we can go right. Check if col of blank is n-1
        if blank_col != self.n - 1:
            moves.append("Right")

        return moves

    def up(self, depth):
        start_idx, end_idx = self.blank_pos, self.blank_pos - self.n

        new_layout = list(self.layout)
        new_layout[end_idx], new_layout[start_idx] = new_layout[start_idx], new_layout[end_idx]

        return State(new_layout, "Up", self, depth)

    def down(self, depth):
        start_idx, end_idx = self.blank_pos, self.blank_pos + self.n

        new_layout = list(self.layout)
        new_layout[end_idx], new_layout[start_idx] = new_layout[start_idx], new_layout[end_idx]

        return State(new_layout, "Down", self, depth)

    def left(self, depth):
        start_idx, end_idx = self.blank_pos, self.blank_pos - 1

        new_layout = list(self.layout)
        new_layout[end_idx], new_layout[start_idx] = new_layout[start_idx], new_layout[end_idx]

        return State(new_layout, "Left", self, depth)

    def right(self, depth):
        start_idx, end_idx = self.blank_pos, self.blank_pos + 1

        new_layout = list(self.layout)
        new_layout[end_idx], new_layout[start_idx] = new_layout[start_idx], new_layout[end_idx]

        return State(new_layout, "Right", self, depth)

    def expand(self, gdepth):
        legal_moves = self.valid_moves()
        for move in legal_moves:
            if move in legal_moves:
                if move == "Up":
                    self.children.append(self.up(gdepth+1))
                if move == "Down":
                    self.children.append(self.down(gdepth+1))
                if move == "Left":
                    self.children.append(self.left(gdepth+1))
                if move == "Right":
                    self.children.append(self.right(gdepth+1))
        return self.children

    def is_goal_state(self):
        return sorted(self.layout) == self.layout


def bfs_search(init_state):
    fringe = []
    visited = []
    extra_table = []

    fringe.append(init_state)
    extra_table.append(hash(init_state))

    max_depth = init_state.depth

    while fringe:
        curr = fringe.pop(0)
        extra_table.remove(hash(curr))
        visited.append(hash(curr))

        if curr.is_goal_state():
            return curr, max_depth

        # expand
        kids = curr.expand(curr.depth)
        for k in kids:
            if k is not None:
                if hash(k) not in extra_table:
                    if hash(k) not in visited:
                        fringe.append(k)
                        extra_table.append(hash(k))
                        if k.depth > max_depth:
                            max_depth = k.depth

# def dfs_search(init_state):


def get_general_moves(final_state):
    moves = []

    curr_state = final_state
    while curr_state.parent is not None:
        moves.append(curr_state.action)
        curr_state = curr_state.parent

    moves.reverse()
    return moves


def main():
    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))

    hard_state = State(begin_state, "Initial", None, 0)

    if sm == "bfs":
        final_state, max_search_depth = bfs_search(hard_state)
        print("Moves ", get_general_moves(final_state))
        print("Search depth ", final_state.depth)
        print("max search depth ", max_search_depth)


if __name__ == "__main__":
    main()
