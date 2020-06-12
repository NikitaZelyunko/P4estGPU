#include "overlap_quads_generate.h"

/*
void getTop(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count);
void getLeft(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count);
void getRight(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count);
void getBottom(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count);
*/
/*
void getTop(unsigned char level, size_t index, unsigned char** masks, sc_array *quads, size_t* res, size_t& res_count);
void getLeft(unsigned char level, size_t index, unsigned char** masks, sc_array *quads, size_t* res, size_t& res_count);
void getRight(unsigned char level, size_t index, unsigned char** masks, sc_array *quads, size_t* res, size_t& res_count);
void getBottom(unsigned char level, size_t index, unsigned char** masks, sc_array *quads, size_t* res, size_t& res_count);
*/

void getTop(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t* res_quads, size_t& res_count);
void getLeft(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t* res_quads, size_t& res_count);
void getRight(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t* res_quads, size_t& res_count);
void getBottom(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t* res_quads, size_t& res_count);

const unsigned char top_bound = 8;
const unsigned char bottom_bound = 4;
const unsigned char right_bound = 2;
const unsigned char left_bound = 1;

//void generate_overlap_p4est_quadrants(test_quad_t *quadrants, int8_t max_level, size_t *start_indexes, unsigned char* boundaries)
void generate_overlap_p4est_quadrants(sc_array *quadrants, int8_t max_level, size_t *&start_indexes, unsigned char* &boundaries, sc_array *res_quadrants)
{
	int n = quadrants->elem_count;
	//int n = 28;

	unsigned char** masks = new unsigned char*[max_level];
	for (int i = 0; i < max_level; i++) {
		masks[i] = new unsigned char[n];
		for (int j = 0; j < n; j++) {
			masks[i][j] = 0;
		}
	}

	unsigned char left_b = 0;
	unsigned char right_b = 1;
	unsigned char left_t = 2;
	unsigned char right_t = 3;

	int prev_i = 0;
	//test_quad_t *prev = &(quadrants[0]);
	p4est_quadrant_t *prev = p4est_quadrant_array_index(quadrants, 0);
	unsigned char prev_level = prev->level;
	/*
	cout << "0_mask: ";
	for (int i = 0; i < max_level; i++) {
		masks[i][prev_i] = left_b;
		cout << (int) masks[i][prev_i] << " ";
	}
	cout << endl;
	*/
	unsigned char prev_position = left_b;
	unsigned char current_position = right_b;
	unsigned char *base_mask = new unsigned char[max_level];

	base_mask[0] = left_b;
	for (int i = 1; i < n; i++) {
		//test_quad_t *current = &(quadrants[i]);
		p4est_quadrant_t *current = p4est_quadrant_array_index(quadrants, i);
		unsigned char cur_level = current->level;
		if (cur_level == prev_level) {
			if (prev_position < right_t) {
				current_position = prev_position + 1;
				for (int j = cur_level - 2; j >= 0; j--) {
					masks[j][i] = masks[j][prev_i];
				}
				masks[cur_level - 1][i] = current_position;
			}
			else {
				current_position = left_b;
				for (int j = cur_level - 2; j >= 0; j--) {
					unsigned char temp = masks[j][prev_i];
					temp = (temp + 1) % 4;
					if (temp) {
						masks[j][i] = temp;
						for (int k = 0; k < j; k++) {
							masks[k][i] = masks[k][prev_i];
						}
						break;
					}
					masks[j][i] = temp;
				}
				masks[cur_level - 1][i] = current_position;
			}
		}
		else {
			for (int j = prev_level - 1; j >= 0; j--) {
				unsigned char temp = masks[j][prev_i];
				temp = (temp + 1) % 4;
				if (temp) {
					masks[j][i] = temp;
					for (int k = 0; k < j; k++) {
						masks[k][i] = masks[k][prev_i];
					}
					break;
				}
				masks[j][i] = temp;
			}
			
			if (cur_level > prev_level) {
				current_position = left_b;
				masks[cur_level - 1][i] = current_position;
			}
			else {
				current_position = masks[cur_level - 1][i];
			}
				
		}
		prev_i = i;
		prev_level = cur_level;
		prev_position = current_position;
		prev = current;
		/*
		cout << i << "_mask: ";
		for (int j = 0; j < max_level; j++) {
			cout << (int) masks[j][i] << " ";
		}
		cout << endl;
		*/
	}

	boundaries = new unsigned char[n];
	size_t quads_count = n;
	for (int i = 0; i < n; i++) {
		unsigned char &bound = (boundaries[i] = 0);
		//unsigned char level = quadrants[i].level;
		unsigned char level = p4est_quadrant_array_index(quadrants, i)->level;
		unsigned char is_bound = 1;
		int j = 1;
		for (j = 1; j < level; j++) {
			if (masks[j][i] % 2) {
				is_bound = 0;
				break;
			}
		}
		if (is_bound) {
			bound |= 1;
			quads_count += 2;
		}else if (j == 1) {
			is_bound = 1;
			for (j = 2; j < level; j++) {
				if (masks[j][i] % 2) {}
				else {
					is_bound = 0;
					break;
				}
			}
			if (is_bound) {
				bound |= 2;
				quads_count += 2;
			}
		}

		is_bound = 1;
		for (j = 1; j < level; j++) {
			if (masks[j][i] > 1) {
				is_bound = 0;
				break;
			}
		}
		if (is_bound) {
			bound |= 4;
			quads_count += 2;
		}
		else if (j == 1) {
			is_bound = 1;
			for (j = 2; j < level; j++) {
				if (masks[j][i] < 2) {
					is_bound = 0;
					break;
				}
			}
			if (is_bound) {
				bound |= 8;
				quads_count += 2;
			}
		}
	}
	/*
	cout << "boundaries:" << endl;
	for (int i = 0; i < n; i++) {
		cout << i<< "_bound_type: " << (int)boundaries[i] << endl;
	}
	*/
	cout << "quads_count: " << quads_count << endl;
	
	start_indexes = new size_t[n];
	size_t *quads_indexes = new size_t[quads_count];
	size_t *res_quads = new size_t[2];
	int j = 0;
	for (int i = 0; i < n; i++) {
		start_indexes[i] = j;
		quads_indexes[j++] = i;
		
		if (boundaries[i] > 0) {
			size_t res_count = 0;
			//unsigned char cur_level = quadrants[i].level;
			p4est_quadrant_t *cur_quad = p4est_quadrant_array_index(quadrants, i);
			unsigned char cur_level = cur_quad->level;
			if (left_bound & boundaries[i]) {
				//getLeft(cur_level, i, masks, boundaries, quadrants, res_quads, res_count);
				/*
				cout << i << "_quad boundaries_left:";
				for (int i = 0; i < res_count; i++) {
				cout << res_quads[i] << " ";
				}
				cout << endl;
				*/
				memcpy(&(quads_indexes[j]), res_quads, res_count * sizeof(size_t));
				j += res_count;
			} else if (right_bound & boundaries[i]) {
				//getRight(cur_level, i, masks, boundaries, quadrants, res_quads, res_count);
				/*
				cout << i << "_quad boundaries_right:";
				for (int i = 0; i < res_count; i++) {
				cout << res_quads[i] << " ";
				}
				cout << endl;
				*/
				memcpy(&(quads_indexes[j]), res_quads, res_count * sizeof(size_t));
				j += res_count;
			}
			if (bottom_bound & boundaries[i]) {
				//getBottom(cur_level, i, masks, boundaries, quadrants, res_quads, res_count);
				/*
				cout << i << "_quad boundaries_bottom:";
				for (int i = 0; i < res_count; i++) {
				cout << res_quads[i] << " ";
				}
				cout << endl;
				*/
				memcpy(&(quads_indexes[j]), res_quads, res_count * sizeof(size_t));
				j += res_count;
			} else if (top_bound & boundaries[i]) {
				//getTop(cur_level, i, masks, boundaries, quadrants, res_quads, res_count);
				/*
				cout << i << "_quad boundaries_top:";
				for (int i = 0; i < res_count; i++) {
				cout << res_quads[i] << " ";
				}
				cout << endl;
				*/
				memcpy(&(quads_indexes[j]), res_quads, res_count * sizeof(size_t));
				j += res_count;
			}
		}
	}
	cout << "quads_collection: " << j << endl;
	/*
	for (int i = 0; i < quads_count; i++) {
		cout << quads_indexes[i] << " ";
	}
	cout << endl;
	system("pause");
	*/
	for(int i = 0; i < max_level; i++) {
		free(masks[i]);
	}
	free(masks);
	free(quads_indexes);
	free(res_quads);
}

const unsigned char top_opposite_table[4] = {2, 3, 0, 1};
const unsigned char bottom_opposite_test[4][4] = {
	{1, 1, 0, 1},
	{1, 1, 1, 0},
	{0, 1, 1, 1},
	{1, 0, 1, 1},
};
const char top_bound_more_level_offset[4] = {0, -1, -2, -3};
const char top_bound_equal_level_offset[4][4] = {
	{2, 1, 0, -1},
	{3, 2, 1, 0},
	{0, -1, -2, -3},
	{1, 0, -1, -2},
};

void getTop(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t res_quads, size_t& res_count)
//void getTop(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count)  
{
	unsigned char start_index = masks[0][index];
	if (start_index < 2) {
		unsigned char left_level_bound = level - 1;
		unsigned char right_level_bound = level + 1;
		unsigned char target_index = masks[level - 1][index];
		unsigned char opposite_index = top_opposite_table[target_index];
		for (int i = index + 1;; i++) {
			if(bottom_bound & boundaries[i]) {
				//unsigned char finded_level = quads[i].level;
				unsigned char finded_level = p4est_quadrant_array_index(quads, i)->level;
				if(finded_level < left_level_bound || finded_level > right_level_bound) {continue;}
				int stop_level = level - 2;
				unsigned char prefix_valid = 1;
				for(int j = 0; j <= stop_level; j++) {
					if(bottom_opposite_test[masks[j][index]][masks[j][i]]) {
						prefix_valid = 0;
						break;
					};
				}
				if(prefix_valid) {
					if(finded_level == left_level_bound) {
						res_indexes[0] = i;
						res_count = 1;
						return;
					} else if( finded_level == level) {
						res_indexes[0] = i + top_bound_equal_level_offset[target_index][masks[finded_level - 1][i]];
						res_count = 1;
						return;
					} else {
						unsigned char finded_level_parent_index = masks[finded_level - 2][i];
						if(finded_level_parent_index == opposite_index) {
							size_t first_index = i + top_bound_more_level_offset[masks[finded_level - 1][i]];
							res_indexes[0] = first_index;
							res_indexes[1] = first_index + 1;
							res_count = 2;
							return;
						}
					}
				}
			}
		}
	}
	res_count = 1;
	res_indexes[0] = -1;
}

const unsigned char bottom_opposite_table[4] = {2, 3, 0, 1};
const unsigned char top_opposite_test[4][4] = {
	{1, 1, 0, 1},
	{1, 1, 1, 0},
	{0, 1, 1, 1},
	{1, 0, 1, 1},
};
const char bottom_bound_more_level_offset[4] = {2, 1, 0, -1};
const char bottom_bound_equal_level_offset[4][4] = {
	{2, 1, 0, -1},
	{3, 2, 1, 0},
	{0, -1, -2, -3},
	{1, 0, -1, -2},
};


void getBottom(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t res_quads, size_t& res_count) 
//void getBottom(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count)
{
	unsigned char start_index = masks[0][index];
	if (start_index > 1) {
		unsigned char left_level_bound = level - 1;
		unsigned char right_level_bound = level + 1;
		unsigned char target_index = masks[level - 1][index];
		unsigned char opposite_index = bottom_opposite_table[target_index];
		for (int i = index - 1;; i--) {
			if(top_bound & boundaries[i]) {
				//unsigned char finded_level = quads[i].level;
				unsigned char finded_level = p4est_quadrant_array_index(quads, i)->level;
				if(finded_level < left_level_bound || finded_level > right_level_bound) {continue;}
				int stop_level = level - 2;
				unsigned char prefix_valid = 1;
				for(int j = 0; j <= stop_level; j++) {
					if(top_opposite_test[masks[j][index]][masks[j][i]]) {
						prefix_valid = 0;
						break;
					};
				}
				if(prefix_valid) {
					if(finded_level == left_level_bound) {
						res_indexes[0] = i;
						res_count = 1;
						return;
					} else if( finded_level == level) {
						res_indexes[0] = i + bottom_bound_equal_level_offset[target_index][masks[finded_level - 1][i]];
						res_count = 1;
						return;
					} else {
						unsigned char finded_level_parent_index = masks[finded_level - 2][i];
						if(finded_level_parent_index == opposite_index) {
							size_t first_index = i + bottom_bound_more_level_offset[masks[finded_level - 1][i]];
							res_indexes[0] = first_index;
							res_indexes[1] = first_index + 1;
							res_count = 2;
							return;
						}
					}
				}
			}
		}
	}
	res_count = 1;
	res_indexes[0] = -1;
}

const unsigned char right_opposite_table[4] = {1, 0, 3, 2};
const unsigned char left_opposite_test[4][4] = {
	{1, 0, 1, 1},
	{0, 1, 1, 1},
	{1, 1, 1, 0},
	{1, 1, 0, 1},
};
const char right_bound_more_level_offset[4] = {0, -1, -2, -3};
const char right_bound_equal_level_offset[4][4] = {
	{1, 0, -1, -2},
	{0, -1, -2, -3},
	{3, 2, 1, 0},
	{2, 1, 0, -1},
};

void getRight(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t res_quads, size_t& res_count) 
//void getRight(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count) 
{
	unsigned char start_index = masks[0][index];
	if (!(start_index % 2)) {
		unsigned char left_level_bound = level - 1;
		unsigned char right_level_bound = level + 1;
		unsigned char target_index = masks[level - 1][index];
		unsigned char opposite_index = right_opposite_table[target_index];
		for (int i = index + 1;; i++) {
			if(left_bound & boundaries[i]) {
				//unsigned char finded_level = quads[i].level;
				unsigned char finded_level = p4est_quadrant_array_index(quads, i)->level;
				if(finded_level < left_level_bound || finded_level > right_level_bound) {continue;}
				int stop_level = level - 2;
				unsigned char prefix_valid = 1;
				for(int j = 0; j <= stop_level; j++) {
					if(left_opposite_test[masks[j][index]][masks[j][i]]) {
						prefix_valid = 0;
						break;
					};
				}
				if(prefix_valid) {
					if(finded_level == left_level_bound) {
						res_indexes[0] = i;
						res_count = 1;
						return;
					} else if( finded_level == level) {
						res_indexes[0] = i + right_bound_equal_level_offset[target_index][masks[finded_level - 1][i]];
						res_count = 1;
						return;
					} else {
						unsigned char finded_level_parent_index = masks[finded_level - 2][i];
						if(finded_level_parent_index == opposite_index) {
							size_t first_index = i + right_bound_more_level_offset[masks[finded_level - 1][i]];
							res_indexes[0] = first_index;
							res_indexes[1] = first_index + 2;
							res_count = 2;
							return;
						}
					}
				}
			}
		}
	}
	res_count = 1;
	res_indexes[0] = -1;
}

const unsigned char left_opposite_table[4] = { 1, 0, 3, 2};
const unsigned char right_opposite_test[4][4] = {
	{1, 0, 1, 1},
	{0, 1, 1, 1},
	{1, 1, 1, 0},
	{1, 1, 0, 1},
};
const char left_bound_more_level_offset[4] = {1, 0, -1, -2};
const char left_bound_equal_level_offset[4][4] = {
	{1, 0, -1, -2},
	{0, -1, -2, -3},
	{3, 2, 1, 0},
	{2, 1, 0, -1},
};

void getLeft(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, sc_array *quads, size_t* res_indexes, p4est_quadrant_t *res_quads, size_t& res_count) 
//void getLeft(unsigned char level, size_t index, unsigned char** masks, unsigned char* boundaries, test_quad_t *quads, size_t* res, size_t& res_count) 
{
	unsigned char start_index = masks[0][index];
	if (start_index % 2) {
		unsigned char left_level_bound = level - 1;
		unsigned char right_level_bound = level + 1;
		unsigned char target_index = masks[level - 1][index];
		unsigned char opposite_index = left_opposite_table[target_index];
		for (int i = index - 1;; i--) {
			if(right_bound & boundaries[i]) {
				//unsigned char finded_level = quads[i].level;
				p4est_quadrant_t *finded_quad = p4est_quadrant_array_index(quads, i);
				unsigned char finded_level = finded_quad->level;
				if(finded_level < left_level_bound || finded_level > right_level_bound) {continue;}
				int stop_level = level - 2;
				unsigned char prefix_valid = 1;
				for(int j = 0; j <= stop_level; j++) {
					if(right_opposite_test[masks[j][index]][masks[j][i]]) {
						prefix_valid = 0;
						break;
					};
				}
				if(prefix_valid) {
					if(finded_level == left_level_bound) {
						res_indexes[0] = i;
						res_count = 1;
						return;
					} else if( finded_level == level) {
						res_indexes[0] = i + left_bound_equal_level_offset[target_index][masks[finded_level - 1][i]];
						res_count = 1;
						return;
					} else {
						unsigned char finded_level_parent_index = masks[finded_level - 2][i];
						if(finded_level_parent_index == opposite_index) {
							size_t first_index = i + left_bound_more_level_offset[masks[finded_level - 1][i]];
							res_indexes[0] = first_index;
							res_indexes[1] = first_index + 2;
							res_count = 2;
							return;
						}
					}
				}
			}
		}
	}
	res_count = 1;
	res_indexes[0] = -1;
}

void test_quads() {
	int n = 28;

	test_quad_t *quads = new quad[n];

	quads[0].level = 2;

	quads[1].level = 2;

	quads[2].level = 2;

	quads[3].level = 3;

	quads[4].level = 3;

	quads[5].level = 3;

	quads[6].level = 3;

	quads[7].level = 2;

	quads[8].level = 2;

	quads[9].level = 2;

	quads[10].level = 2;

	quads[11].level = 2;

	quads[12].level = 2;

	quads[13].level = 3;

	quads[14].level = 3;

	quads[15].level = 3;

	quads[16].level = 3;

	quads[17].level = 3;

	quads[18].level = 3;

	quads[19].level = 3;

	quads[20].level = 3;

	quads[21].level = 3;

	quads[22].level = 3;

	quads[23].level = 3;

	quads[24].level = 3;

	quads[25].level = 2;

	quads[26].level = 2;

	quads[27].level = 2;

	size_t *start_indexes;
	unsigned char *boundaries;

	//generate_overlap_p4est_quadrants(quads, 3, start_indexes, boundaries);
}
