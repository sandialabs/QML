import numpy as np
import pickle
import scipy.sparse

def main():
    # file_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/albertii.ani.txt'  # Replace with the path to your TSV file
    file_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/esch/esch50.ani'  # Replace with the path to your TSV file
    threshold = 50.0

    # color_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/albertii.meta.txt'  
    color_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/esch/esch.meta' 
    color_to_int_mapping = {}
    
    species = "Escherichia__coli_D"
    genome_to_species = {}

    # try:
    with open(color_path, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            # print("tokens: ", tokens)

            # check species
            # print("specis", tokens[4], tokens)
            # if (tokens[4] == species):
            genome_to_species[tokens[0]] = tokens[4]
                # print("specis", tokens[4], tokens)


    string_to_int_mapping = {}
    int_to_string_mapping = {}
    current_int = 1
    
    encoded_data = []
    listVals = []
    removeNodes = []

    try:
        with open(file_path, 'r') as file:
            print("file open")
            for line in file:
                tokens = line.strip().split('\t')
                encoded_line = []
                
                # check species
                # print("specis", tokens[0], genome_to_species[tokens[0]])
                if (genome_to_species[tokens[0]] == species):
                    val = 1.0
                    sizeScale = 1.0
                    justLabels = 0
                    alignment = 0
                    smallerNode = 0
                    # print("alignment ", float(tokens[3]))
                    if (float(tokens[3]) > threshold):
                        for token in tokens:
                            justLabels += 1
                            if (justLabels < 3):
                                if token in string_to_int_mapping:
                                    encoded_line.append(string_to_int_mapping[token])
                                else:
                                    print(token, current_int)
                                    print("specis", tokens[0], genome_to_species[tokens[0]])
                                    string_to_int_mapping[token] = current_int
                                    int_to_string_mapping[current_int] = token
                                    # int_to_string_mapping[current_int] = token
                                    encoded_line.append(current_int)
                                    current_int += 1
                                if (justLabels == 1):
                                    sizeScale = float(token.split('.')[1])
                                    smallerNode = encoded_line[0]
                                elif (justLabels == 2):
                                    sizeScale /= float(token.split('.')[1])
                                    if (sizeScale > 1.0):
                                        smallerNode = encoded_line[1]
                                        sizeScale = 1 / sizeScale
                            else:
                                val *= float(token) / 100.0
                                alignment = float(token)
                        # print("tokens ", tokens)
                        # print("val scale ", val, sizeScale)
                        # print("align ", alignment)
                        if (alignment > threshold):
                            # print("added edge", sizeScale < 0.25, val > 0.99)
                            # remove node if small island of existing data
                            if (sizeScale < 0.25 and val > 0.99):
                                if smallerNode not in removeNodes:
                                    removeNodes.append(smallerNode)
                                # print("Node removed")
                            val *= sizeScale
                            listVals.append(val)
                            encoded_data.append(encoded_line)
                    


    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred:", e)

    # re assign indecies from removed nodes
    removeNodes.sort()
    removeID = 0
    maxID = 0
    for sublist in encoded_data:
        for number in sublist:
            if number > maxID:
                maxID = number
    # maxID = max(sublist[-1] for sublist in encoded_data)
    print("maxID before", maxID)
    newIndex = np.zeros((maxID+1,1))
    new_string_to_int_mapping = {}
    newIDcount = 1
    # print("removeNodes ", removeNodes)
    for i in range(1,maxID+1):
        if (removeID <= len(removeNodes)-1):
            if (removeNodes[removeID] == i):
                newIndex[i] = -1
                removeID += 1
                # print("removeID", removeID)
            else:
                newIndex[i] = newIDcount
                newIDcount += 1
        else:
            newIndex[i] = newIDcount
            newIDcount += 1
    reduced_data = []
    reducedVals = []
    i = 0
    # print("encoded_data size", len(encoded_data), encoded_data)
    for list in encoded_data:
        # print("encoded data", list[0], list)
        if (newIndex[list[0]] != -1 and newIndex[list[0]] != -1):
            # print("data ", encoded_data[i][0], encoded_data[i][1])
            reduced_data.append([int(newIndex[encoded_data[i][0]].item()), int(newIndex[encoded_data[i][1]].item())])
            reducedVals.append(listVals[i])
            i += 1
        else:
            print("removed ", i)
    print("int to stirng map ", int_to_string_mapping)
    for i in range(1, 1+maxID):
        if (newIndex[i] != -1):
            print("i ", i, newIndex[i])
            new_string_to_int_mapping[int_to_string_mapping[i]] = newIndex[i][0]
    # print("data", encoded_data, reduced_data)
    # print("vals ", listVals)
    # print("r vals", reducedVals)
    print("remove nodes ",  removeNodes)
    print("maxID", maxID)
    print("string to int ", string_to_int_mapping)
    print("\n \n \n", new_string_to_int_mapping)
    listVals = reducedVals
    encoded_data = reduced_data
    string_to_int_mapping = new_string_to_int_mapping


 
    
    maxID = max(encoded_data)[0]
    if (max(encoded_data)[1] > maxID):
        maxID = max(encoded_data)[1]
    print("maxID", maxID)


    # color_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/albertii.meta.txt'  
    color_path = '/Users/pnooste/Documents/extra/data/snlBioAlbertii/esch/esch.meta'  
    
    maxID = max(encoded_data)[0]
    if (max(encoded_data)[1] > maxID):
        maxID = max(encoded_data)[1]
    print("maxID", maxID)

    color_to_int_mapping = {}
    int_to_color_mapping = {}
    color_int = 1
    color_data = np.zeros((maxID,1))
    color_data2 = np.zeros((maxID,1))

    # try:
    with open(color_path, 'r') as file:
        for line in file:
            tokens = line.strip().split('\t')
            encoded_line = []
            # print("tokens: ", tokens)
            if (genome_to_species[tokens[0]] == species):
                justLabels = 0
                for token in tokens:
                    justLabels += 1
                    if (justLabels > 1):
                        if token in color_to_int_mapping:
                            encoded_line.append(color_to_int_mapping[token])
                        else:
                            # print(token, color_int)
                            color_to_int_mapping[token] = color_int
                            int_to_color_mapping[color_int] = token
                            encoded_line.append(color_int)
                            color_int += 1
                try:
                    # print("string index ", string_to_int_mapping[tokens[0]]-1, color_to_int_mapping[tokens[1]])
                    color_data[int(string_to_int_mapping[tokens[0]]-1)] = color_to_int_mapping[tokens[1]]
                    color_data2[int(string_to_int_mapping[tokens[0]]-1)] = color_to_int_mapping[tokens[2]]
                except:
                    continue
        # except FileNotFoundError:
        #     print("File not found.")
        # except Exception as e:
        #     print("An error occurred:", e)

    
    
    # A = np.matrix(np.ones((maxID,maxID))*np.inf)
    # A = scipy.sparse.csr_matrix((maxID, maxID))
    A = scipy.sparse.lil_matrix((maxID, maxID))
    # A = np.zeros((maxID,maxID))
    countE = 0
    for edge in encoded_data:
        # print("edge ", edge)
        A[edge[0]-1, edge[1]-1] = listVals[countE]
        A[edge[0]-1, edge[1]-1] = ((1-(A[edge[0]-1, edge[1]-1])) + 0.1) * 10
        # print(i, j, (1-(data[i,j]) + 0.1)*10, file=file)
        countE += 1

    colorVec = np.zeros((maxID,1))
    colorVec2 = np.zeros((maxID,1))
    # print("color_data", color_data)
    # print("color len ", len(color_data))
    # print("colorMap: ", color_data)
    for i in range(maxID):
        colorVec[i] = color_data[i]
        colorVec2[i] = color_data2[i]

    print("colors ", color_data.size, color_data)

    # Now you can use the encoded_data list and the updated string_to_int_mapping dictionary
    # print("Encoded data:", encoded_data)
    # print("weights", listVals)
    # print("String to int mapping:", string_to_int_mapping)

    filename = 'bioGraph.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(A,file)

    filename = 'bioGraphColored.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(color_data,file)

    filename = 'bioGraphColored2.pickle'
    with open(filename, 'wb') as file:
        pickle.dump(color_data2,file)

    filename = 'bioGraphLabels.pickle'
    with open(filename, 'wb') as file:
        print(int_to_color_mapping)
        pickle.dump(int_to_color_mapping, file)

if __name__ == "__main__":
    main()



