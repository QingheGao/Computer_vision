import numpy as np

def construct_surface(p, q, path_type='average'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """

        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    height_map[i, j] =0
                elif j==0:
                    height_map[i, j] = height_map [i-1,j]+q[i,j]
                else:
                    height_map[i, j] = height_map[i, j-1] + p[i, j]



    elif path_type=='row':
        """
        ================
        Your code here
        ================
        """
        for i in range(h):
            for j in range(w):
                if i == 0 and j == 0:
                    height_map[i, j] = 0
                elif i == 0:
                    height_map[i, j] = height_map[i, j - 1] + p[i, j]
                else:
                    height_map[i, j] = height_map[i-1 , j] + q[i, j]

    elif path_type=='average':
        """
        ================
        Your code here
        ================
        """
        left = construct_surface(p, q, path_type='column')
        right = construct_surface(p, q, path_type='row')
        height_map = (left + right)/2

        
    return height_map
        
