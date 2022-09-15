

import numpy as np

# Calculate the euclidian distance in n-space of the route r traversing cities
#c, ending at the path start. path_distance = lambda r,c:
# np.sum([np.linalg.norm(c[r[p]]-c[r[p-1]]) for p in range(len(r))]) Reverse the
# order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))


def two_opt(xs,ys,improvement_threshold): 
    # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    
    xx, yy = np.meshgrid(xs,ys)
    dd = np.sqrt( (xx-xx.T)**2 + (yy-yy.T)**2)
    
    #print(dd)
    
    route = np.arange(dd.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    #best_distance = path_distance(route,cities) # Calculate the distance of the initial path.
    best_distance = np.sum(dd[route[:-1],route[1:]])

    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!

        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
    
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
        
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
            
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                #print(new_route)
                #new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                new_distance = np.sum(dd[new_route[:-1],new_route[1:]])
                
                #print("%0.3f, %0.3f" % (new_distance, best_distance))
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
            # print("d:%0.3f, if: %0.3f" % (best_distance,improvement_factor))
        
    return route # When the route is no longer improving substantially, stop searching and return the route.


def distance_from_route(xs,ys,route):
    ''' Calculates the distance of the path given by the xs and ys coordinates in the order specified by route'''
    
    xx = np.array(xs)[route]
    yy = np.array(ys)[route]
    d = np.sum(np.sqrt( np.diff(xx)**2 + np.diff(yy)**2))
    return d

def two_opt_lines(x1s,y1s,x2s,y2s,improvement_threshold):
    
    # TODO: Make sure these are not too big for dd to fit in memory.

    # Interleaves coordinates for each point. 
    x = np.array([[a,b] for a,b in zip(x1s,x2s)]).flatten()
    y = np.array([[a,b] for a,b in zip(y1s,y2s)]).flatten()
    
    xx, yy = np.meshgrid(x,y)
    
    # Pre calculate the distance between all pairs of points. 
    dd = np.sqrt( (xx-xx.T)**2 + (yy-yy.T)**2)

    # Along the diagonal of dd, is the length of each line, 
    # interleaved with the distance from the end of one line
    # to the next. Off diagnoal values give the distance from
    # the start/end of one line to the start of the next. 
    # 
    
    # If we only consider traveling along the lines in the direction specified,
    # Then valid route indices would indicate only the first point in the line,
    # meaning [0,2,4...]. Then the distance from the start of one to the start of 
    # the next would be the length of the line plus the distance from the end of the 
    # line to the start of the next. This would be indexed as 
    #
    # length_of_the_line = d[route[i],route[i]+1]
    # end_of_line_to_start_of_the_next = d[route[i]+1,route[i+1]]
    #
    # If we want to consider going either direction along a line...
    # We would need a route index that considers all the points...
    # But a route from p1 to p2 where p1 and p2 are the endpooints of the same 
    # line is invalid. 
    
    # If we want only go backward along the line, then the route indices are [1,3,5,7...]
    # and the distance frmo the start of one to the start of the next would be
    #
    # length_of_the_line = d[route[i],route[i]-1]
    # end_of_line_to_start_of_the_next = d[route[i]-1,route[i+1]]
    
    # But this will give redundancies, because we are including every line segment
    # in both directions and we don't want to go cover the same line segment twice.
    
        
    route = np.arange(dd.shape[0]) # Make an array of row numbers corresponding to cities.
    improvement_factor = 1 # Initialize the improvement factor.
    
    
    even = np.argwhere(np.mod(route,2) == 0) + 1
    odd = np.argwhere(np.mod(route,2)==1) - 1
    
    # BUG The indexing and logic here is incorrect as we try to sum up the path segments for 
    # a given route.
    best_distance = np.sum( dd[0,route[even[0]]] + dd[route[even[:-1]],route[even[:-1]+1]] + dd[route[even[:-1]+1],route[even[:-1]+2]])
    best_distance += np.sum(dd[0,route[odd[0]]] + dd[route[odd[1:]],route[odd[1:]-1]] + dd[route[odd[1:]-1],route[odd[1:]+1]])
 
    #best_distance = np.sum(dd[route[:-1],route[1:]])
  
    while improvement_factor > improvement_threshold: # If the route is still improving, keep going!

        distance_to_beat = best_distance # Record the distance at the beginning of the loop.
    
        for swap_first in range(1,len(route)-2): # From each city except the first and last,
        
            for swap_last in range(swap_first+1,len(route)): # to each of the cities following,
            
                new_route = two_opt_swap(route,swap_first,swap_last) # try reversing the order of these cities
                print(new_route)
                #new_distance = path_distance(new_route,cities) # and check the total distance with this modification.
                new_distance = np.sum( dd[0,1] + dd[route[even[:-1]],route[even[:-1]+1]] + dd[route[even[:-1]+1],route[even[:-1]+2]])
                new_distance += np.sum(dd[0,1] + dd[route[odd[1:]],route[odd[1:]-1]] + dd[route[odd[1:]-1],route[odd[1:]+1]])
                
                print("%0.3f, %0.3f" % (new_distance, best_distance))
                if new_distance < best_distance: # If the path distance is an improvement,
                    route = new_route # make this the accepted best route
                    best_distance = new_distance # and update the distance corresponding to this route.
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.
            # print("d:%0.3f, if: %0.3f" % (best_distance,improvement_factor))
        
    return route # When the route is no longer improving substantially, stop searching and return the route.
