use crate::decode::ImageRegion;
use crate::frame::{FrameEncoding, FrameHeader};
use crate::frame_data::{FrameData, FrameSectionKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctFrameMetadata {
    pub width: u32,
    pub height: u32,
    pub group_dim: u32,
    pub groups_x: u32,
    pub groups_y: u32,
    pub dc_groups_x: u32,
    pub dc_groups_y: u32,
    pub sections: Vec<VarDctSectionMetadata>,
    pub ac_groups: Vec<VarDctGroupMetadata>,
    pub dc_groups: Vec<VarDctGroupMetadata>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarDctSectionMetadata {
    pub section_logical_id: usize,
    pub section_physical_index: usize,
    pub section_kind: FrameSectionKind,
    pub codestream_offset: usize,
    pub payload_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarDctGroupMetadata {
    pub group: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl VarDctFrameMetadata {
    pub fn ac_groups_intersecting_region(&self, region: ImageRegion) -> Vec<usize> {
        self.ac_groups
            .iter()
            .filter(|group| group_intersects_region(group, region))
            .map(|group| group.group)
            .collect()
    }
}

pub fn read_vardct_frame_metadata(
    frame_header: &FrameHeader,
    frame_data: &FrameData,
) -> Option<VarDctFrameMetadata> {
    if frame_header.encoding != FrameEncoding::VarDct {
        return None;
    }

    let sections = frame_data
        .sections
        .iter()
        .map(|section| VarDctSectionMetadata {
            section_logical_id: section.logical_id,
            section_physical_index: section.physical_index,
            section_kind: section.kind,
            codestream_offset: section.codestream_offset,
            payload_size: section.size,
        })
        .collect();

    Some(VarDctFrameMetadata {
        width: frame_header.frame_size.width,
        height: frame_header.frame_size.height,
        group_dim: frame_header.group_layout.group_dim,
        groups_x: frame_header.group_layout.groups_x,
        groups_y: frame_header.group_layout.groups_y,
        dc_groups_x: frame_header.group_layout.dc_groups_x,
        dc_groups_y: frame_header.group_layout.dc_groups_y,
        sections,
        ac_groups: group_metadata(
            frame_header.group_layout.groups_x,
            frame_header.group_layout.groups_y,
            frame_header.group_layout.group_dim,
            frame_header.frame_size.width,
            frame_header.frame_size.height,
        ),
        dc_groups: group_metadata(
            frame_header.group_layout.dc_groups_x,
            frame_header.group_layout.dc_groups_y,
            frame_header.group_layout.dc_group_dim,
            frame_header.frame_size.width,
            frame_header.frame_size.height,
        ),
    })
}

fn group_metadata(
    groups_x: u32,
    groups_y: u32,
    group_dim: u32,
    frame_width: u32,
    frame_height: u32,
) -> Vec<VarDctGroupMetadata> {
    let mut groups = Vec::with_capacity(groups_x as usize * groups_y as usize);
    for gy in 0..groups_y {
        for gx in 0..groups_x {
            let x = gx * group_dim;
            let y = gy * group_dim;
            groups.push(VarDctGroupMetadata {
                group: groups.len(),
                x,
                y,
                width: group_dim.min(frame_width.saturating_sub(x)),
                height: group_dim.min(frame_height.saturating_sub(y)),
            });
        }
    }
    groups
}

fn group_intersects_region(group: &VarDctGroupMetadata, region: ImageRegion) -> bool {
    let Some(group_right) = group.x.checked_add(group.width) else {
        return true;
    };
    let Some(group_bottom) = group.y.checked_add(group.height) else {
        return true;
    };
    let Some(region_right) = region.x.checked_add(region.width) else {
        return true;
    };
    let Some(region_bottom) = region.y.checked_add(region.height) else {
        return true;
    };
    group.x < region_right
        && region.x < group_right
        && group.y < region_bottom
        && region.y < group_bottom
}
