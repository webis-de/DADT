import xml.etree.ElementTree as ET

def readTrainingFile(filename):
	document_contents = {}
	document_authors = {}
	author_ids = {}
	document_ids = {}
	
	root = ET.parse(filename).getroot()
	
	if root.tag != 'training':
		return ()
		
	for document_node in root:
		# Get Nodes
		author_node = document_node[0]
		content_node = document_node[1]
		
		# Get Values
		document_name = document_node.get("file")
		author_name = author_node.get("id")
		document_content = content_node.text
		
		# Assign Unique IDs
		document_id = len(document_ids)
		if author_name in author_ids:
			author_id = author_ids[authorName]
		else:
			author_id = len(a)
			
		# Store Inforamtion
		document_ids[document_id] = document_name
		document_authors[document_id] = [author_id]
		author_ids[author_id] = author_name
		document_contents[document_id] = document_content
		
	return (document_contents, document_authors, author_ids, document_ids)
		
def testFile(filename):
	document_contents = {}
	document_ids = {}
	
	root = ET.parse(filename).getroot()
	
	if root.tag != 'training':
		return ()
		
	for document_node in root:
		# Get Nodes
		content_node = document_node[1]
		
		# Get Values
		document_name = document_node.get("file")
		document_content = content_node.text
		
		# Assign Unique IDs
		document_id = len(document_ids)
			
		# Store Inforamtion
		document_ids[document_id] = document_name
		document_contents[document_id] = document_content
	
	return (document_contents, document_authors, author_ids, document_ids)
	
print(readTrainingFile('../../pan11-author-identification-training-corpus-2011-04-08/SmallTrainEscaped.xml'))
